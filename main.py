# ============================================================================
# 1. Завантажити дані за особистим вподабанням, але щоб одни і затрибутів
# був час (рекомендація : курси валют, економічні показники в часі, кліматичні
# показники, і нші)
# ============================================================================

import pandas as pd

# Завантаження середньомісячних температур у Ноттінгемі.
url = (
    "https://vincentarelbundock.github.io/"
    "Rdatasets/csv/datasets/nottem.csv"
)
data = pd.read_csv(url)

# Переведення температури з градусів Фаренгейта у градуси Цельсія.
df = pd.DataFrame({
    "Дата": pd.date_range(
        start="1920-01-01",
        periods=len(data),
        freq="MS"
    ),
    "Температура, °C": (data["value"].to_numpy() - 32) * 5 / 9
})

# Перевірка та збереження даних.
assert len(df) == 240, "Неочікувана кількість спостережень."
assert not df.isna().any().any(), "У даних є пропуски."

df.to_csv(
    "nottingham_temperature.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Набір даних: Nottem")
print("Місце спостережень: Ноттінгем, Велика Британія")
print("Показник: середньомісячна температура повітря")
print("Одиниця вимірювання: °C")
print("Атрибут часу: Дата")
print("Періодичність: один місяць")
print(f"Кількість спостережень: {len(df)}")
print(
    f"Період: {df['Дата'].min():%m.%Y}"
    f" — {df['Дата'].max():%m.%Y}"
)
print(f"Пропущених значень: {df.isna().sum().sum()}")

print("\nПерші 12 спостережень:")
print(df.head(12).round(2).to_string(index=False))




# ============================================================================
# 2. Побудувати графік ряду
# ============================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    df["Дата"],
    df["Температура, °C"],
    color="steelblue",
    linewidth=1.5
)

plt.title("Середньомісячна температура повітря в Ноттінгемі (1920–1939)")
plt.xlabel("Рік")
plt.ylabel("Температура, °C")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("02_temperature_series.png", dpi=300)
plt.show()


# ============================================================================
# 3. Виконати згладження ряду (обрати два методи і їх порівняти)
# ============================================================================

# Метод 1: ковзне середнє за 3 місяці.
df["Ковзне середнє"] = (
    df["Температура, °C"]
    .rolling(window=3)
    .mean()
)

# Метод 2: експоненціальне згладжування.
df["Експоненціальне згладжування"] = (
    df["Температура, °C"]
    .ewm(alpha=0.3, adjust=False)
    .mean()
)

# Побудова графіка.
plt.figure(figsize=(14, 6))

plt.plot(
    df["Дата"],
    df["Температура, °C"],
    label="Початковий ряд",
    color="gray",
    alpha=0.5,
    linewidth=1
)

plt.plot(
    df["Дата"],
    df["Ковзне середнє"],
    label="Ковзне середнє за 3 місяці",
    color="royalblue",
    linewidth=1.5
)

plt.plot(
    df["Дата"],
    df["Експоненціальне згладжування"],
    label="Експоненціальне згладжування (α = 0,3)",
    color="darkorange",
    linewidth=1.5
)

plt.title("Порівняння двох методів згладжування температури")
plt.xlabel("Рік")
plt.ylabel("Температура, °C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("03_smoothing_comparison.png", dpi=300)
plt.show()

# Виведення значень для порівняння.
columns = [
    "Дата",
    "Температура, °C",
    "Ковзне середнє",
    "Експоненціальне згладжування"
]

print("Порівняння значень за перші 12 місяців:")
print(df[columns].head(12).round(2).to_string(index=False))




# ============================================================================
# 4. Зробити прогноз як приклад на 12 періодів вперед
# ============================================================================

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Часовий ряд із місячною частотою.
series = df.set_index("Дата")["Температура, °C"].asfreq("MS")

# Модель із сезонністю, яка повторюється кожні 12 місяців.
model = ExponentialSmoothing(
    series,
    trend=None,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

# Прогноз на наступні 12 місяців.
forecast = model.forecast(12)

forecast_table = pd.DataFrame({
    "Місяць": forecast.index.strftime("%Y-%m"),
    "Прогноз температури, °C": forecast.to_numpy()
})

print("Прогноз середньомісячної температури на 1940 рік:")
print(forecast_table.round(2).to_string(index=False))

# Збереження прогнозу.
forecast_table.to_csv(
    "04_forecast_12_months.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================================
# 5. Дослідити сезонність і побудувати сезонну складову
# ============================================================================

from statsmodels.tsa.seasonal import seasonal_decompose

# Розкладання часового ряду на складові.
decomposition = seasonal_decompose(
    series,
    model="additive",
    period=12
)

# Сезонна складова для кожного місяця року.
seasonal_by_month = (
    decomposition.seasonal
    .groupby(decomposition.seasonal.index.month)
    .mean()
)

months = [
    "Січень", "Лютий", "Березень", "Квітень",
    "Травень", "Червень", "Липень", "Серпень",
    "Вересень", "Жовтень", "Листопад", "Грудень"
]

plt.figure(figsize=(12, 5))

plt.plot(
    range(1, 13),
    seasonal_by_month.to_numpy(),
    marker="o",
    color="seagreen",
    linewidth=2
)

plt.axhline(0, color="gray", linestyle="--")
plt.xticks(range(1, 13), months, rotation=45)
plt.title("Сезонна складова температури в Ноттінгемі")
plt.xlabel("Місяць")
plt.ylabel("Сезонна складова, °C")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("05_seasonal_component.png", dpi=300)
plt.show()

print("Сезонна складова за місяцями:")
print(pd.DataFrame({
    "Місяць": months,
    "Сезонна складова, °C": seasonal_by_month.to_numpy()
}).round(2).to_string(index=False))


# ============================================================================
# 6. Дослідити випадковість і побудувати випадкову складову (за наявності)
# ============================================================================

from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = decomposition.resid.dropna()

plt.figure(figsize=(12, 5))

plt.plot(
    residuals.index,
    residuals,
    color="slateblue",
    linewidth=1
)

plt.axhline(0, color="black", linestyle="--")
plt.title("Залишкова (випадкова) складова температури")
plt.xlabel("Рік")
plt.ylabel("Відхилення, °C")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("06_residual_component.png", dpi=300)
plt.show()

# Перевірка залежності залишків на лагах від 1 до 12 місяців.
test = acorr_ljungbox(residuals, lags=[12], return_df=True)
p_value = test["lb_pvalue"].iloc[0]

print(f"Середнє значення залишків: {residuals.mean():.2f} °C")
print(f"Стандартне відхилення залишків: {residuals.std():.2f} °C")
print(f"p-значення тесту Льюнга—Бокса: {p_value:.4f}")

if p_value > 0.05:
    print(
        "Тест не виявив значущої автокореляції "
        "на лагах до 12 місяців."
    )
else:
    print(
        "Тест виявив автокореляцію на лагах до 12 місяців. "
        "Залишки не можна вважати повністю випадковими."
    )


# ============================================================================
# 7. Побудувати графік фактичних і прогнозних значень.
# ============================================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    series.iloc[-36:],
    label="Фактична температура",
    color="royalblue",
    marker="o",
    markersize=4
)

plt.plot(
    forecast,
    label="Прогноз на 1940 рік",
    color="darkorange",
    linestyle="--",
    marker="o",
    markersize=4
)

plt.axvline(
    forecast.index[0],
    color="gray",
    linestyle=":",
    label="Початок прогнозу"
)

plt.title("Фактична температура та прогноз на 12 місяців")
plt.xlabel("Рік")
plt.ylabel("Температура, °C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("07_actual_and_forecast.png", dpi=300)
plt.show()

