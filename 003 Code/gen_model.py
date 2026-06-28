
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

class GenModel:
    COMP = ['우분%', '돈분%', '커피박%', '톱밥%', '슬러지%']
    COMP_NAMES = ['우분', '돈분', '커피박', '톱밥', '슬러지']
    GAS = ['CO2', 'CO', 'CH4', 'N2O', 'NH3', 'H2S', 'NOx', 'SO2']
    G4 = ['CO2g/kg', 'COg/kg', 'CH4g/kg', 'N2Og/kg', 'NH3g/kg', 'H2Sg/kg', 'NOxg/kg', 'SO2g/kg']
    TREE = ['수율(%)', '고정탄소(%)', 'H/C비', '회분(%)']
    LIN = ['발열량(cal/g)', 'BET(m²/g)', '벌크밀도(g/cm³)', '탄화점(°C)', '함수율(%)']

    def __init__(self):
        self.prop_models = {}
        self.gas_models = {}
        self.is_trained = False

    def train(self, prop_csv, gas_csv, verbose=True):
        prop = pd.read_csv(prop_csv)
        X = prop[self.COMP + ['탄화온도(°C)']].values
        for c in self.TREE:
            self.prop_models[c] = ExtraTreesRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=10,
                random_state=42, n_jobs=-1).fit(X, prop[c].values)
        for c in self.LIN:
            self.prop_models[c] = LinearRegression().fit(X, prop[c].values)
        self.PROP = self.TREE + self.LIN

        C = pd.read_csv(gas_csv, skiprows=2, encoding='utf-8-sig')
        C.columns = [c.replace('\n', '').strip() for c in C.columns]
        tr = C[C['연도'] <= 2021]
        Xtr = self._enc(tr)
        for g4, g in zip(self.G4, self.GAS):
            self.gas_models[g] = LinearRegression().fit(
                Xtr, pd.to_numeric(tr[g4], errors='coerce'))

        self.is_trained = True
        if verbose:
            print('친구 모델 학습 완료 (물성 9 + 가스 8)')
        return self

    def _enc(self, d):
        return np.column_stack([
            d[self.COMP].astype(float).values,
            np.sin(2 * np.pi * d['월'] / 12),
            np.cos(2 * np.pi * d['월'] / 12),
        ])

    @staticmethod
    def grade_ibi(fc, hc, bet, ash, mo):
        if fc > 80 and hc < 0.40 and bet > 400 and ash < 8 and mo < 8:
            return 'Premium(S)', 350000
        if fc >= 60 and hc <= 0.60 and bet >= 200 and ash < 12 and mo < 10:
            return 'A급', 220000
        if fc >= 40 and hc <= 0.70 and bet >= 50 and ash < 15 and mo < 12:
            return 'B급', 150000
        if fc >= 30 and hc <= 0.85 and bet >= 10 and ash < 20 and mo < 15:
            return 'C급', 100000
        return '등급외', 0

    @staticmethod
    def grade_ebc(fc, hc, bet, ash, mo):
        if fc > 75 and hc < 0.45 and bet > 350 and ash < 10 and mo < 10:
            return 'EBC Premium', 300000
        if fc >= 55 and hc <= 0.65 and bet >= 150 and ash < 14 and mo < 12:
            return 'EBC Seal', 200000
        if fc >= 40 and hc <= 0.75 and bet >= 50 and ash < 16 and mo < 12:
            return 'EBC Feed', 130000
        return '등급외', 0

    @staticmethod
    def bio_srf_grade(hhv_calg, moisture, ash):
        
        checks = {
            '발열량(LHV≥3000)': hhv_calg >= 3000,
            '수분(≤25%)': moisture <= 25,
            '회분(≤15%)': ash <= 15,
        }
        fails = [k for k, ok in checks.items() if not ok]
        return ('적합(예비)' if not fails else '부적합'), fails, checks

    def predict(self, comp, temp, month):
        
        cv = [comp[n] for n in self.COMP_NAMES]
        if abs(sum(cv) - 100) > 1e-6:
            return {'error': ['조성비 합=100']}

        xp = np.array([cv + [temp]], float)
        xg = np.column_stack([np.array([cv], float),
                              [np.sin(2 * np.pi * month / 12)],
                              [np.cos(2 * np.pi * month / 12)]])
        p = {c: float(self.prop_models[c].predict(xp)[0]) for c in self.PROP}
        for g in self.GAS:
            p[g] = max(0.0, float(self.gas_models[g].predict(xg)[0]))
        co2e = p['CO2'] + 28 * p['CH4'] + 265 * p['N2O']

        fc, hc, bet, ash, mo = p['고정탄소(%)'], p['H/C비'], p['BET(m²/g)'], p['회분(%)'], p['함수율(%)']
        ig, ip = self.grade_ibi(fc, hc, bet, ash, mo)
        eg, ep = self.grade_ebc(fc, hc, bet, ash, mo)
        if ip == 0 and ep == 0:
            best_g, best_p = '등급외', 0
        elif ip >= ep:
            best_g, best_p = f'IBI {ig}', ip
        else:
            best_g = eg if eg.startswith('EBC') else f'EBC {eg}'
            best_p = ep

        srf_v, srf_fails, srf_checks = self.bio_srf_grade(p['발열량(cal/g)'], mo, ash)

        return {
            '입력': {
                '조성(%)': comp, '탄화온도(°C)': temp, '월': month,
            },
            '물성 예측': {
                '수율(%)': round(p['수율(%)'], 2),
                '고정탄소(%)': round(fc, 2),
                'H/C비': round(hc, 4),
                '회분(%)': round(ash, 2),
                '발열량(cal/g)': round(p['발열량(cal/g)'], 1),
                'BET(m²/g)': round(bet, 1),
                '탄화점(°C)': round(p['탄화점(°C)'], 1),
                '함수율(%)': round(mo, 2),
            },
            '가스 배출 (g/kg)': {g: round(p[g], 4) for g in self.GAS},
            '탄소·배출': {
                'CO2e 합계 (g/kg)': round(co2e, 2),
            },
            '품질등급': {
                'IBI 등급': ig,
                'EBC 등급': eg,
                '최적등급': best_g,
                '톤당가격(원)': best_p,
            },
            '고형연료 (바이오SRF·비성형)': {
                '판정': srf_v,
            },
        }
