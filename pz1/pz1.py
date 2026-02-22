import logging.config
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import List

from scipy.signal import find_peaks



class PlotterAnalyze:
    
    @staticmethod
    def create_plot(theta: np.ndarray,
                    zero_grad: np.ndarray,
                    half_max: np.ndarray,
                    max_grad: np.ndarray,
                    peaks: List[float],
                    max_grad_numb: float = 45,
                    title: str = "Диаграмма направленности ЛФАР") -> None:
        """
        Строит нормированные амплитудные диаграммы направленности
        в декартовой системе координат (в dB).

        Параметры:
            theta : ndarray
                массив углов в радианах

            zero_grad : ndarray
                ДН при theta = 0°

            half_max : ndarray
                ДН при theta = theta_max/2

            max_grad : ndarray
                ДН при theta = theta_max

            peaks : List[float]
                Пики боковых лепестков theta = 0, max/2, max
            
            max_grad_numb : float
                Максимальное значение угла в градусах
                
            title : str
                заголовок графика
        """

        theta_zero = [np.deg2rad(0), np.deg2rad(max_grad_numb/2), np.deg2rad(max_grad_numb)]
        F0_dB = 20 * np.log10(np.maximum(zero_grad, 1e-12))
        F_half_dB = 20 * np.log10(np.maximum(half_max, 1e-12))
        F_max_dB = 20 * np.log10(np.maximum(max_grad, 1e-12))

        fig = plt.figure(figsize=(12, 10))

        ax1 = fig.add_subplot(2, 2, 1, projection='polar')
        ax1.plot(theta, F0_dB)
        ax1.set_title("θ₀ = 0°")

        ax2 = fig.add_subplot(2, 2, 2, projection='polar')
        ax2.plot(theta, F_half_dB)
        ax2.set_title(f"θ₀ = {(max_grad_numb/2):.1f}°")

        ax3 = fig.add_subplot(2, 2, 3, projection='polar')
        ax3.plot(theta, F_max_dB)
        ax3.set_title(f"θ₀ = {max_grad_numb}°")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(theta_zero, peaks)
        ax4.set_xlabel("θ₀")
        ax4.set_ylabel("SLL (dB)")
        ax4.set_title("Зависимость уровня бокового лепестка от θ₀")
        ax4.grid(True)

        fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.show()


class RCSSolver:
    def __init__(self, 
                 freq: float, 
                 theta_max: float, 
                 delta_theta: float):
        self.freq = freq
        self.theta_max = np.deg2rad(theta_max)
        self.delta_theta = np.deg2rad(delta_theta)
        self.lmbd = 3e8 / self.freq
        self.d_max = self.lmbd / (1 + np.sin(self.theta_max))
        self.k = 2 * np.pi / self.lmbd
        self.theta = np.linspace(-np.pi/2, np.pi/2, 2000)


    @property
    def N(self) -> int:
        return int(np.ceil((np.degrees(51)*self.lmbd)
                           /(self.d*np.rad2deg(self.delta_theta)))) 
    

    @property
    def d(self) -> float:
        return self.lmbd / (1 + np.sin(self.theta_max)) * 0.99


    def _func(self, theta0: float) -> np.ndarray:
        F = np.zeros_like(self.theta, dtype=complex)

        for n in range(self.N):
            x_n = n * self.d
            F += (np.exp(1j * -self.k * x_n * np.sin(theta0)) 
                  * np.exp(1j * self.k * x_n * np.sin(self.theta)))
        
        F *= np.sqrt(np.cos(self.theta)) 

        return F


    def _norm(self, func: np.ndarray) -> np.ndarray:
        func_abs = np.abs(func)
        return func_abs/np.max(func_abs)
    

    def _find_sll(self, func: np.ndarray) -> float:
        
        peaks, _ = find_peaks(func)

        peak_values = func[peaks]
        sorted_peaks = np.sort(peak_values)

        sll_linear = sorted_peaks[-2]
        sll_db = 20 * np.log10(sll_linear)

        return sll_db


    def solver(self) -> dict:
        """
    Выполняет расчёт нормированных амплитудных диаграмм направленности
    линейной фазированной антенной решётки.

    Принцип действия:
    - Геометрия решётки (число элементов N и шаг d) определяется
      по заданному максимальному углу сканирования theta_max.
    - Для трёх значений угла фазового возбуждения theta
      (0°, theta_max/2 и theta_max) вычисляется комплексная сумма полей
      всех элементов решётки.
    - Учитывается амплитудная характеристика одиночного элемента
      f1(theta) = sqrt(cos theta).
    - Полученная диаграмма направленности нормируется
      к максимальному значению.

    Возвращает:
        dict:
        {
            "theta": ndarray
                Массив углов theta в рад
            
            "zero_grad": ndarray
                нормированная ДН при theta = 0°,

            "half_max": ndarray
                нормированная ДН при theta = theta_max/2,

            "max_grad": ndarray
                нормированная ДН при theta = theta_max
        }"""
        F0 = self._norm(self._func(theta0=0))
        F_half_max = self._norm(self._func(theta0=(self.theta_max / 2)))
        F_max = self._norm(self._func(theta0=self.theta_max))
        
        logging.info("Расчёт завершён. Параметры решётки:")
        print(f"N: {self.N}\nd: {self.d}\nlambda: {self.lmbd}")

        logging.info("Нахождение наибольших лепестков для каждого угла фазового возбуждения:")
        F0_peak = self._find_sll(F0)
        F_half_max_peak = self._find_sll(F_half_max)
        F_max_peak = self._find_sll(F_max)
        print(f"zero_grad: {F0_peak}\nhalf_max_grad: {F_half_max_peak}\nmax_grad: {F_max_peak}")

        return {
            "theta": self.theta,
            "zero_grad": F0,
            "half_max": F_half_max,
            "max_grad": F_max,
            "peaks": [F0_peak, F_half_max_peak, F_max_peak]
        }

    
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=r"[%(levelname)s] %(message)s"
    )

    solver = RCSSolver(freq=1e10, 
                       theta_max=np.degrees(45),
                       delta_theta=np.degrees(15))
    data = solver.solver()

    PlotterAnalyze.create_plot(theta=data["theta"],
                               zero_grad=data["zero_grad"],
                               half_max=data["half_max"],
                               max_grad=data["max_grad"],
                               peaks=data["peaks"])