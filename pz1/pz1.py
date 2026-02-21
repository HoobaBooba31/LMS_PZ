import matplotlib.pyplot as plt

import numpy as np


class PlotterAnalyze:
    
    @staticmethod
    def create_plot(theta: np.ndarray,
                    zero_grad: np.ndarray,
                    half_max: np.ndarray,
                    max_grad: np.ndarray,
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

            db_min : float
                нижний предел по оси Y (дБ)

            title : str
                заголовок графика
        """


        F0_dB = 20 * np.log10(np.maximum(zero_grad, 1e-12))
        F_half_dB = 20 * np.log10(np.maximum(half_max, 1e-12))
        F_max_dB = 20 * np.log10(np.maximum(max_grad, 1e-12))

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), 
                                             subplot_kw={'projection': 'polar'})

        # Построение графиков
        ax0.plot(theta, F0_dB, label="θ₀ = 0°")
        ax1.plot(theta, F_half_dB, label="θ₀ = θmax/2")
        ax2.plot(theta, F_max_dB, label="θ₀ = θmax")

        for ax, label in zip([ax0, ax1, ax2], 
                              ["θ₀ = 0°", "θ₀ = θmax/2", "θ₀ = θmax"]):
            ax.set_title(label)
            ax.grid(True)
            ax.legend(loc='lower left')

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
        print(f"N: {self.N}\nd: {self.d}\nlambda: {self.lmbd}")

        return {
            "theta": self.theta,
            "zero_grad": F0,
            "half_max": F_half_max,
            "max_grad": F_max,
        }

    
if __name__ == "__main__":
    solver = RCSSolver(freq=1e10, 
                       theta_max=np.degrees(45),
                       delta_theta=np.degrees(15))
    data = solver.solver()

    PlotterAnalyze.create_plot(theta=data["theta"],
                               zero_grad=data["zero_grad"],
                               half_max=data["half_max"],
                               max_grad=data["max_grad"])