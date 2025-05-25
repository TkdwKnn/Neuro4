import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 27

df = pd.read_csv('dataset_simple.csv')
print("Данные:")
print(df.head())
print(f"Размер данных: {df.shape}")


class NNet_Classification(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()
        )

    def forward(self, X):
        pred = self.layers(X)
        return pred
X = torch.tensor(df.iloc[:, [0, 1]].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32).reshape(-1, 1)

print(f"\nРазмерность признаков X: {X.shape}")
print(f"Размерность меток y: {y.shape}")
print(f"Первые 5 примеров X:\n{X[:5]}")
print(f"Первые 5 меток y: {y[:5].flatten()}")


X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X_normalized = (X - X_mean) / X_std

print(f"\nНормализованные данные X (первые 5 примеров):\n{X_normalized[:5]}")


inputSize = X.shape[1]
hiddenSizes = 5
outputSize = 1

print(f"\nПараметры сети:")
print(f"Входной слой: {inputSize} нейронов")
print(f"Скрытый слой: {hiddenSizes} нейронов")
print(f"Выходной слой: {outputSize} нейрон")

net = NNet_Classification(inputSize, hiddenSizes, outputSize)

print(f"\nПараметры сети:")
for name, param in net.named_parameters():
    print(f"{name}: {param.shape}")

lossFn = nn.BCELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

print(f"\nНачинаем обучение...")
epochs = 200

for i in range(epochs):
    pred = net.forward(X_normalized)

    loss = lossFn(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (i + 1) % 20 == 0:
        print(f'Ошибка на {i + 1} итерации: {loss.item():.4f}')

print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")

with torch.no_grad():
    pred = net.forward(X_normalized)

    pred_classes = (pred >= 0.5).float()

    accuracy = (pred_classes == y).float().mean()
    print(f"Точность классификации: {accuracy.item():.4f} ({accuracy.item() * 100:.1f}%)")

    errors = (pred_classes != y).sum().item()
    total = len(y)
    print(f"Количество ошибок: {errors} из {total}")

print(f"\nДетальный анализ предсказаний:")
print("Возраст | Доход  | Истина | Вероятность | Предсказание | Правильно?")

with torch.no_grad():
    pred = net.forward(X_normalized)
    pred_classes = (pred >= 0.5).float()

    for i in range(len(X)):
        age = X[i, 0].item()
        income = X[i, 1].item()
        true_class = int(y[i].item())
        probability = pred[i].item()
        predicted_class = int(pred_classes[i].item())
        correct = "✓" if true_class == predicted_class else "✗"

        print(f"{age:6.0f} | {income:6.0f} | {true_class:6} | {probability:11.3f} | {predicted_class:12} | {correct}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['red' if label == 0 else 'blue' for label in y.flatten()]
plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Исходные данные\n(красный = не купит, синий = купит)')
plt.grid(True)

# График 2: Предсказания сети
plt.subplot(1, 2, 2)
with torch.no_grad():
    pred = net.forward(X_normalized)
    pred_classes = (pred >= 0.5).float()

colors_pred = ['red' if pred == 0 else 'blue' for pred in pred_classes.flatten()]
plt.scatter(X[:, 0], X[:, 1], c=colors_pred, alpha=0.7)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Предсказания сети\n(красный = не купит, синий = купит)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nОбучение завершено!")