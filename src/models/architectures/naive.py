import torch
import torch.nn as nn
import torch.nn.functional as F

class NaivePredictor(nn.Module):
    def __init__(self, forecast_horizon, method="1"):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.method = method
        
        valid_methods = ("1", "S", "2", "mean")
        assert method in valid_methods, f"method must be in {valid_methods}"
        
    def _check_seasonality(self, x, m_list):
        """Performs a 90% autocorrelation test per sample."""
        batch_size, window_size = x.shape
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        
        is_seasonal = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        
        for i in range(batch_size):
            m = int(m_list[i].item())
            if m <= 1 or window_size < 2 * m:
                continue
            
            # ACF at lag m for this specific sample
            top = (x_centered[i, m:] * x_centered[i, :-m]).sum()
            bottom = (x_centered[i]**2).sum()
            acf = top / (bottom + 1e-8)
            
            threshold = 1.645 / (window_size**0.5)
            is_seasonal[i] = acf.abs() > threshold
        return is_seasonal

    def _get_ma_2xm(self, x_row, m):
        """
        Computes a 2xM moving average for a single 1D tensor row.
        Matches standard statistical software implementations for classical decomposition.
        """
        window_size = x_row.shape[0]
        # Step 1: Classical M-order moving average
        p1 = (m // 2, (m - 1) // 2)
        padded1 = F.pad(x_row.view(1, 1, -1), p1, mode='replicate')
        ma_m = F.avg_pool1d(padded1, kernel_size=m, stride=1).view(-1)
        
        # Step 2: 2-order moving average to center it perfectly
        padded2 = F.pad(ma_m.view(1, 1, -1), (1, 0), mode='replicate')
        ma_2xm = F.avg_pool1d(padded2, kernel_size=2, stride=1).view(-1)
        return ma_2xm

    def _get_normalized_seasonal_indices(self, x_row, ma_row, m):
        """
        Extracts seasonal components, averages them across cycles, 
        and normalizes them to sum up/average out to 1.0.
        """
        window_size = x_row.shape[0]
        # Multiplicative raw seasonal components
        raw_seasonal = x_row / (ma_row + 1e-8)
        
        # Aggregate across cycles
        seasonal_cycle = torch.zeros(m, device=x_row.device)
        for period in range(m):
            # Gather indices belonging to the same seasonal period (e.g., all Mondays)
            indices = torch.arange(period, window_size, step=m, device=x_row.device)
            seasonal_cycle[period] = raw_seasonal[indices].mean()
            
        # Normalize indices so their mean is exactly 1.0 (multiplicative standard)
        seasonal_cycle = seasonal_cycle / (seasonal_cycle.mean() + 1e-8)
        return seasonal_cycle

    def forward(self, x, exogs):
        batch_size, window_size = x.shape

        # Map exogs to seasonal periods
        # exogs: [[18], [6], [8]] -> m: [12, 1, 4]
        m_list = torch.zeros_like(exogs)
        m_list[exogs == 6] = 1
        m_list[exogs == 8] = 4
        m_list[exogs == 18] = 12
        m_list = m_list.squeeze(1)

        if self.method == "1":
            return x[:, -1:] .expand(-1, self.forecast_horizon)

        elif self.method == "mean":
            return x.mean(dim=1, keepdim=True).expand(-1, self.forecast_horizon)

        elif self.method == "S":
            forecast = torch.zeros(batch_size, self.forecast_horizon, device=x.device)
            for i in range(batch_size):
                m = int(m_list[i].item())
                if window_size < m or m <= 1:
                    forecast[i] = x[i, -1:].expand(self.forecast_horizon)
                else:
                    last_season = x[i, -m:]
                    repeats = (self.forecast_horizon // m) + 1
                    forecast[i] = last_season.repeat(repeats)[:self.forecast_horizon]
            return forecast

        elif self.method == "2":
            is_seasonal = self._check_seasonality(x, m_list)
            forecast = torch.zeros(batch_size, self.forecast_horizon, device=x.device)
            
            for i in range(batch_size):
                m = int(m_list[i].item())
                
                if is_seasonal[i] and m > 1 and window_size >= 2 * m:
                    # 1. Smooth out trend via MA_(2xm)
                    ma_trend = self._get_ma_2xm(x[i], m)
                    
                    # 2. Get normalized seasonal cycle factors
                    seasonal_cycle = self._get_normalized_seasonal_indices(x[i], ma_trend, m)
                    
                    # 3. Deseasonalize the historical series to find x_t
                    # Replicate seasonal cycle across the history to match window_size
                    historical_repeats = (window_size // m) + 1
                    full_historical_seasonality = seasonal_cycle.repeat(historical_repeats)[:window_size]
                    
                    # Target the final adjusted observation: x_T = y_T / s_T
                    x_T = x[i, -1] / (full_historical_seasonality[-1] + 1e-8)
                    
                    # 4. Roll out seasonal factors into the future horizon
                    # The formula maps the horizon to the cycle starting right after T
                    # We can align it by continuing the seasonal sequence from index T onwards
                    full_future_seasonality = seasonal_cycle.repeat(historical_repeats + (self.forecast_horizon // m) + 1)
                    # Slice out the exact horizon steps matching T to T+h
                    horizon_seasonality = full_future_seasonality[window_size : window_size + self.forecast_horizon]
                    
                    # 5. Multiply the final naive value by the future horizon seasonal components
                    forecast[i] = x_T * horizon_seasonality
                    
                else:
                    # Default strictly back to standard Naive 1 if not seasonal
                    forecast[i] = x[i, -1:].expand(self.forecast_horizon)
                    
            return forecast

        return None

def model_naive(ws, fh, naive_type='2'):
    model = NaivePredictor(forecast_horizon=fh, method=naive_type)
    return {
        'model': model,
        'id': f"naive_{naive_type}",
        'window_size': ws,
        'forecasting_horizon': fh,
        'n_exogs': 1,
    }
