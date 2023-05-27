import numpy
from scipy import signal
import matplotlib.pyplot as plt


class SignalProcessor:

    def __init__(self, sample_size, sample_rate, max_frequency, filter_frequency, quantization_levels):
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.max_frequency = max_frequency
        self.filter_frequency = filter_frequency
        self.quantization_levels = quantization_levels

    def process_signals(self):
        random_normal = numpy.random.normal(0, 10, self.sample_size)
        time_samples = numpy.arange(self.sample_size) / self.sample_rate
        cutoff_frequency_ratio = self.max_frequency / (self.sample_rate / 2)
        filter_params = signal.butter(3, cutoff_frequency_ratio, "low", output="sos")
        filtered_signal = signal.sosfiltfilt(filter_params, random_normal)

        quantized_signals = []
        dispersions = []
        snr_values = []

        for levels in self.quantization_levels:
            quantized_signal, quantize_table, bits = self.quantize_signal(filtered_signal, levels)
            quantized_signals.append(quantized_signal)
            self.plot_quantization_table(levels, quantize_table, "Значення сигналу", "Кодова послідовність",
                                         f"Таблиця квантування для {levels} рівнів")
            self.plot_signal_sequence(numpy.arange(0, len(bits)), bits, "Біти", "Амплітуда сигналу",
                                      f"Кодова послідовність сигналу при кількості рівнів квантування {levels}")
            dispersion, snr = self.calculate_dispersion_and_snr(filtered_signal, quantized_signal)
            dispersions.append(dispersion)
            snr_values.append(snr)

        self.plot_signals(time_samples, quantized_signals, "Час (секунди)", "Амплітуда сигналу",
                          f"Цифрові сигнали з рівнями квантування (4, 16, 64, 256)")
        self.plot_values(self.quantization_levels, dispersions, "Кількість рівнів квантування", "Дисперсія",
                                     "Залежність дисперсії від кількості рівнів квантування")
        self.plot_values(self.quantization_levels, snr_values, "Кількість рівнів квантування", "ССШ",
                                     "Залежність співвідношення сигнал-шум від кількості рівнів квантування")

    def quantize_signal(self, filtered_signal, levels):
        bits = []
        delta = (numpy.max(filtered_signal) - numpy.min(filtered_signal)) / (levels - 1)
        quantized_signal = delta * numpy.round(filtered_signal / delta)
        quantize_levels = numpy.arange(numpy.min(quantized_signal), numpy.max(quantized_signal) + 1, delta)
        bit_sequences = numpy.arange(0, levels)
        bit_sequences = [format(bits, '0' + str(int(numpy.log(levels) / numpy.log(2))) + 'b') for bits in bit_sequences]
        quantize_table = numpy.c_[quantize_levels[:levels], bit_sequences[:levels]]
        for signal_value in quantized_signal:
            for index, value in enumerate(quantize_levels[:levels]):
                if numpy.round(numpy.abs(signal_value - value), 0) == 0:
                    bits.append(bit_sequences[index])
                    break
        bits = [int(item) for item in list(''.join(bits))]
        return quantized_signal, quantize_table, bits

    def calculate_dispersion_and_snr(self, original_signal, quantized_signal):
        error = quantized_signal - original_signal
        dispersion = numpy.var(error)
        snr = numpy.var(original_signal) / dispersion
        return dispersion, snr

    def plot_quantization_table(self, levels, quantize_table, column_label_first, column_label_second, title):
        fig, ax = plt.subplots(figsize=(14 / 2.54, levels / 2.54))
        table = ax.table(cellText=quantize_table, colLabels=[column_label_first, column_label_second], loc="center")
        table.set_fontsize(14)
        table.scale(1, 2)
        ax.axis('off')
        fig.savefig(f"figures\\{title}.png", dpi=600)
        plt.show()

    def plot_signal_sequence(self, x, y, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax.step(x, y, linewidth=0.1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.title(title, fontsize=14)
        fig.savefig(f"figures\\{title}.png", dpi=600)
        plt.show()

    def plot_signals(self, x, y, x_label, y_label, title):
        fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
        s = 0
        for i in range(0, 2):
            for j in range(0, 2):
                ax[i][j].plot(x, y[s], linewidth=1)
                s += 1
        fig.supxlabel(x_label, fontsize=14)
        fig.supylabel(y_label, fontsize=14)
        fig.suptitle(title, fontsize=14)
        fig.savefig(f"figures\\{title}.png", dpi=600)
        plt.show()

    def plot_values(self, x, y, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax.plot(x, y, linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.title(title, fontsize=14)
        fig.savefig(f"figures\\{title}.png", dpi=600)
        plt.show()


if __name__ == '__main__':
    processor = SignalProcessor(500, 1000, 13, 20, [4, 16, 64, 256])
    processor.process_signals()
