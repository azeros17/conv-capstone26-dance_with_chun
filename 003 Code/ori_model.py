
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

class OriModel:

    STAGE_TEMP_MAP = {
        '초기건조': (200, 250), '초기열분해': (275, 300),
        '셀룰로오스분해': (325, 350), '주탄화': (375, 400),
        '고온탄화': (425, 500), '안정화': (525, 600),
        '고급탄화': (625, 700), '최고탄화': (725, 800),
    }
    EMISSION_THRESHOLD = 1750.0
    GWP = {'CO2': 1, 'CH4': 28, 'N2O': 265}

    IBI_GRADES = [
        ('Premium(S)', 80, 0.40, 400, 8,  8,  350000),
        ('A급',        60, 0.60, 200, 12, 10, 220000),
        ('B급',        40, 0.70, 50,  15, 12, 150000),
        ('C급',        30, 0.85, 10,  20, 15, 100000),
    ]
    EBC_GRADES = [
        ('EBC Premium', 75, 0.45, 350, 10, 10, 300000),
        ('EBC Seal',    55, 0.65, 150, 14, 12, 200000),
        ('EBC Feed',    40, 0.75, 50,  16, 12, 130000),
    ]

    COMP_NAMES = ['우분', '돈분', '커피박', '톱밥', '슬러지']
    COMP_B = ['우분%', '돈분%', '커피박%', '톱밥%', '슬러지%']
    GAS_COLS = ['CO2g/kg','COg/kg','CH4g/kg','N2Og/kg','NH3g/kg','H2Sg/kg','NOxg/kg','SO2g/kg']
    B_TARGETS = ['수율(%)', '고정탄소(%)', 'H/C비', '회분(%)']

    SUPP_TARGETS = ['발열량(cal/g)', '함수율(%)']

    def __init__(self):
        self.model_B = None
        self.model_C = None
        self.supp_models = {}
        self.supp_feat = {}
        self.model_charpoint = None
        self.model_bet = None
        self.is_trained = False

    def train(self, b_csv, c_csv, supp_csv, verbose=True):
        self._train_B(b_csv)
        self._train_C(c_csv)
        self._train_supplements(supp_csv)
        self.is_trained = True
        if verbose:
            self._report()
        return self

    def _train_B(self, b_csv):
        B = pd.read_csv(b_csv, skiprows=2, encoding='utf-8-sig')
        B.columns = [c.replace('\n', '').strip() for c in B.columns]
        B = B[B['DS'] == 'B'].reset_index(drop=True)
        X = B[self.COMP_B + ['탄화온도(°C)']].astype(float).values
        Y = B[self.B_TARGETS].apply(pd.to_numeric, errors='coerce').values
        self.model_B = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self.model_B.fit(X, Y)
        self._B_train_score = self.model_B.score(X, Y)

    def _train_C(self, c_csv):
        C = pd.read_csv(c_csv, skiprows=2, encoding='utf-8-sig')
        C.columns = [c.replace('\n', '').strip() for c in C.columns]
        train_C = C[C['연도'] <= 2021].copy()
        X = self._encode_C(train_C)
        Y = train_C[self.GAS_COLS].apply(pd.to_numeric, errors='coerce').values
        self.model_C = LinearRegression().fit(X, Y)
        test_C = C[C['연도'] == 2022].copy()
        from sklearn.metrics import r2_score
        pred = self.model_C.predict(self._encode_C(test_C))
        Yt = test_C[self.GAS_COLS].apply(pd.to_numeric, errors='coerce').values
        self._C_test_r2 = {g: r2_score(Yt[:, i], pred[:, i]) for i, g in enumerate(self.GAS_COLS)}

    def _train_supplements(self, supp_csv):
        from sklearn.metrics import r2_score
        df = pd.read_csv(supp_csv, skiprows=3, encoding='utf-8-sig')
        df.columns = [str(c).replace('\n', '').replace(' ', '').strip() for c in df.columns]
        df = df.dropna(subset=[df.columns[0]])
        df = df[df[df.columns[0]].astype(str).str.match(r'^\d+$', na=False)].reset_index(drop=True)
        g = lambda c: pd.to_numeric(df[c], errors='coerce').values
        FC, HC, YL, ASH = g('고정탄소(%)'), g('H/C비'), g('수율(%)'), g('회분(%)')
        pool = {0: FC, 1: HC, 2: YL, 3: ASH}
        feat_map = {
            '발열량(cal/g)':  [0],
            '함수율(%)':      [0, 1],
        }
        col_map = {
            '발열량(cal/g)': '발열량(cal/g)',
            '함수율(%)': '함수율(%)',
        }
        self._supp_r2 = {}
        self._supp_fc_range = (float(np.nanmin(FC)), float(np.nanmax(FC)))
        for tgt in self.SUPP_TARGETS:
            idx = feat_map[tgt]
            Xp = np.column_stack([pool[i] for i in idx])
            y = g(col_map[tgt])
            m = LinearRegression().fit(Xp, y)
            self.supp_models[tgt] = m
            self.supp_feat[tgt] = idx
            self._supp_r2[tgt] = r2_score(y, m.predict(Xp))

        comp_cols = ['우분%', '돈분%', '커피박%', '톱밥%', '슬러지%']
        X_comp = df[comp_cols].astype(float).values
        y_cp = g('탄화점(°C)')
        self.model_charpoint = ExtraTreesRegressor(100, random_state=42, n_jobs=-1).fit(X_comp, y_cp)
        self._supp_r2['탄화점(°C)'] = r2_score(y_cp, self.model_charpoint.predict(X_comp))

        X_hc = HC.reshape(-1, 1)
        y_bet = g('BET(m²/g)')
        self.model_bet = ExtraTreesRegressor(100, random_state=42, n_jobs=-1).fit(X_hc, y_bet)
        self._supp_r2['BET(m²/g)'] = r2_score(y_bet, self.model_bet.predict(X_hc))

    @staticmethod
    def _encode_C(df):
        sin_m = np.sin(2 * np.pi * df['월'] / 12).values
        cos_m = np.cos(2 * np.pi * df['월'] / 12).values
        comp = df[['우분%', '돈분%', '커피박%', '톱밥%', '슬러지%']].astype(float).values
        return np.column_stack([comp, sin_m, cos_m])

    def validate_input(self, comp, temp, month):
        errors = []
        total = sum(comp.values())
        if abs(total - 100.0) > 0.5:
            errors.append(f"조성 합 100% 아님: {total:.2f}%")
        for k, v in comp.items():
            if v < 0:
                errors.append(f"{k} 음수: {v}")
        if not (200 <= temp <= 800):
            errors.append(f"온도 범위 초과(200~800): {temp}")
        if not (1 <= month <= 12):
            errors.append(f"월 범위 초과(1~12): {month}")
        return errors

    @classmethod
    def temp_to_stage(cls, temp):
        for stage, (lo, hi) in cls.STAGE_TEMP_MAP.items():
            if lo <= temp <= hi:
                return stage
        return 'unknown'

    @staticmethod
    def carbon_seq(yield_pct, fixed_c):
        return yield_pct * fixed_c / 10000.0

    @staticmethod
    def co2e_seq(c_seq):
        return c_seq * 44.0 / 12.0

    @classmethod
    def _grade_by_table(cls, table, fc, hc, bet, ash, mo):
        
        for name, fc_lo, hc_hi, bet_lo, ash_hi, mo_hi, price in table:
            if fc >= fc_lo and hc < hc_hi and bet >= bet_lo and ash < ash_hi and mo < mo_hi:
                return name, price
        return '등급외', 0

    @classmethod
    def ibi_grade(cls, fc, hc, bet, ash, mo):
        return cls._grade_by_table(cls.IBI_GRADES, fc, hc, bet, ash, mo)

    @classmethod
    def ebc_grade(cls, fc, hc, bet, ash, mo):
        return cls._grade_by_table(cls.EBC_GRADES, fc, hc, bet, ash, mo)

    @classmethod
    def ks_fit(cls, fc, hc, bet, ash, mo, sludge):
        
        if sludge > 15:
            return '❌ 슬러지>15%'
        if fc > 30 and hc < 0.7 and bet > 10 and ash < 15 and mo < 12:
            return '✅ 적합'
        return '❌ 부적합'

    @classmethod
    def best_grade(cls, ibi, ibi_p, ebc, ebc_p):
        
        if ibi_p == 0 and ebc_p == 0:
            return '등급외', 0
        if ibi_p >= ebc_p:
            return f'IBI {ibi}', ibi_p
        label = ebc if ebc.startswith('EBC') else f'EBC {ebc}'
        return label, ebc_p

    @staticmethod
    def bio_srf_grade(hhv_calg, moisture, ash):
        
        checks = {
            '발열량(LHV≥3000)': hhv_calg >= 3000,
            '수분(≤25%)': moisture <= 25,
            '회분(≤15%)': ash <= 15,
        }
        fails = [k for k, ok in checks.items() if not ok]
        verdict = '적합(예비)' if not fails else '부적합'
        return verdict, fails, checks

    @classmethod
    def co2e_sum(cls, gases):
        return (gases['CO2g/kg'] + gases['CH4g/kg'] * cls.GWP['CH4']
                + gases['N2Og/kg'] * cls.GWP['N2O'])

    @classmethod
    def emission_grade(cls, avg_co2e):
        return '고' if avg_co2e >= cls.EMISSION_THRESHOLD else '중'

    def comp_avg_co2e(self, comp):
        months = np.arange(1, 13)
        sin_a, cos_a = np.sin(2*np.pi*months/12), np.cos(2*np.pi*months/12)
        cv = np.array([comp[n] for n in self.COMP_NAMES])
        X = np.column_stack([np.tile(cv, (12, 1)), sin_a, cos_a])
        gp = self.model_C.predict(X)
        co2e = gp[:, 0] + gp[:, 2]*self.GWP['CH4'] + gp[:, 3]*self.GWP['N2O']
        return float(co2e.mean())

    def predict(self, comp, temp, month):
        if not self.is_trained:
            raise RuntimeError("train() 먼저 호출")
        errors = self.validate_input(comp, temp, month)
        if errors:
            return {'error': errors}

        xb = np.array([[comp['우분'], comp['돈분'], comp['커피박'],
                        comp['톱밥'], comp['슬러지'], temp]])
        yield_pct, fixed_c, hc, ash = self.model_B.predict(xb)[0]

        pool = {0: fixed_c, 1: hc, 2: yield_pct, 3: ash}
        CLIP = {'발열량(cal/g)': (0, 7000), '함수율(%)': (0, 100)}
        supp = {}
        for tgt in self.SUPP_TARGETS:
            idx = self.supp_feat[tgt]
            xv = np.array([[pool[i] for i in idx]])
            raw = float(self.supp_models[tgt].predict(xv)[0])
            lo, hi = CLIP[tgt]
            supp[tgt] = min(max(raw, lo), hi)

        x_comp = np.array([[comp['우분'], comp['돈분'], comp['커피박'],
                            comp['톱밥'], comp['슬러지']]])
        char_point = float(self.model_charpoint.predict(x_comp)[0])

        bet_v = float(self.model_bet.predict(np.array([[hc]]))[0])
        supp['BET(m²/g)'] = bet_v

        fc_lo, fc_hi = self._supp_fc_range
        supp_extrapolated = not (fc_lo <= fixed_c <= fc_hi)

        sin_m, cos_m = np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
        xc = np.array([[comp['우분'], comp['돈분'], comp['커피박'],
                        comp['톱밥'], comp['슬러지'], sin_m, cos_m]])
        gases = dict(zip(self.GAS_COLS, self.model_C.predict(xc)[0]))

        c_seq = self.carbon_seq(yield_pct, fixed_c)
        co2e_seq_v = self.co2e_seq(c_seq)
        co2e_total = self.co2e_sum(gases)
        avg_co2e = self.comp_avg_co2e(comp)
        emiss = self.emission_grade(avg_co2e)

        mo_v = supp['함수율(%)']
        ibi, ibi_p = self.ibi_grade(fixed_c, hc, bet_v, ash, mo_v)
        ebc, ebc_p = self.ebc_grade(fixed_c, hc, bet_v, ash, mo_v)
        ks = self.ks_fit(fixed_c, hc, bet_v, ash, mo_v, comp['슬러지'])
        best_g, best_p = self.best_grade(ibi, ibi_p, ebc, ebc_p)

        srf_verdict, srf_fails, srf_checks = self.bio_srf_grade(
            supp['발열량(cal/g)'], mo_v, ash)

        return {
            '입력': {
                '조성(%)': comp, '탄화온도(°C)': temp,
                '공정단계(자동)': self.temp_to_stage(temp), '월': month,
            },
            '물성 예측 (10개)': {
                '수율(%)': round(yield_pct, 2),
                '고정탄소(%)': round(fixed_c, 2),
                'H/C비': round(hc, 4),
                '회분(%)': round(ash, 2),
                '발열량(cal/g)': round(supp['발열량(cal/g)'], 1),
                'BET(m²/g)': round(supp['BET(m²/g)'], 1),
                '탄화점(°C)': round(char_point, 1),
                '함수율(%)': round(supp['함수율(%)'], 2),
            },
            '가스 배출 (g/kg)': {k: round(v, 4) for k, v in gases.items()},
            '품질등급 (9시트 기준)': {
                'IBI 등급': ibi,
                'EBC 등급': ebc,
                'KS 농진청': ks,
                '최적등급': best_g,
                '톤당가격(원)': best_p,
            },
            '고형연료 (바이오SRF·비성형)': {
                '판정': srf_verdict,
            },
            '탄소·배출 (룰베이스)': {
                '탄소격리(tC/tBC)': round(c_seq, 4),
                'CO2e 격리(t/tBC)': round(co2e_seq_v, 4),
                'CO2e 합계 (입력월, g/kg)': round(co2e_total, 2),
                'CO2e 12개월 평균': round(avg_co2e, 2),
                '배출등급': emiss,
            },
            '신뢰도 경고': (
                '등급판정에 쓰인 보충물성(BET·함수율)이 외삽 — 고정탄소 %.1f%%가 '
                '9시트 학습범위(%.0f~%.0f%%) 밖. 등급·보충값 신뢰도 낮음(참고용).'
                % (fixed_c, fc_lo, fc_hi)
            ) if supp_extrapolated else None,
        }

    def _report(self):
        print("=" * 60)
        print("[학습 완료] 10개 물성 통합 파이프라인")
        print("=" * 60)
        print(f"\n● 모델 B (조성+온도→4물성) ExtraTrees")
        print(f"    train R² (4타깃 평균) = {self._B_train_score:.4f}")
        print(f"    └ GroupKFold 검증 R²: 수율0.96·고정탄소0.96·H/C0.97·회분0.97")
        print(f"\n● 모델 C (조성+월→가스8종) Linear, 2022 test R²")
        for g, r in self._C_test_r2.items():
            print(f"    {g:10s}: {r:.4f}")
        print(f"\n● 보충 상관식 (모델B출력→5물성) 9시트 n=15 적합 R²")
        for t, r in self._supp_r2.items():
            flag = "  ⚠ 신뢰도 낮음(참고용)" if r < 0.75 else ""
            print(f"    {t:14s}: {r:.4f}{flag}")
        print(f"\n● 룰베이스: IBI등급·탄소격리·CO2e·배출등급 (공식)")

if __name__ == '__main__':
    pipe = OriModel().train(
        b_csv='3_탄화온도_B.csv',
        c_csv='4_가스연속_C.csv',
        supp_csv='9_원료배합별_바이오차_비료.csv',
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("시나리오 테스트")
    print("=" * 60)
    scenarios = [
        ('저온 — 우분위주',   {'우분': 50, '돈분': 20, '커피박': 0, '톱밥': 30, '슬러지': 0}, 250, 3),
        ('중온 — 균형',       {'우분': 40, '돈분': 0, '커피박': 20, '톱밥': 30, '슬러지': 10}, 450, 6),
        ('고온 — 톱밥위주',   {'우분': 30, '돈분': 0, '커피박': 20, '톱밥': 40, '슬러지': 10}, 700, 9),
        ('최고온',            {'우분': 40, '돈분': 0, '커피박': 25, '톱밥': 25, '슬러지': 10}, 800, 12),
    ]
    for name, comp, temp, month in scenarios:
        r = pipe.predict(comp, temp, month)
        print(f"\n--- {name} (온도{temp}°C, {month}월, 단계:{r['입력']['공정단계(자동)']}) ---")
        p = r['물성 예측 (10개)']
        print(f"  수율 {p['수율(%)']}% | 고정탄소 {p['고정탄소(%)']}% | H/C {p['H/C비']} | 회분 {p['회분(%)']}%")
        print(f"  발열량 {p['발열량(cal/g)']} cal/g | BET {p['BET(m²/g)']} m²/g | 탄화점 {p['탄화점(°C)']}°C | 함수율 {p['함수율(%)']}%")
        q = r['품질등급 (9시트 기준)']
        print(f"  IBI: {q['IBI 등급']} | EBC: {q['EBC 등급']} | KS: {q['KS 농진청']}")
        print(f"  최적등급: {q['최적등급']} | 톤당가격 {q['톤당가격(원)']:,}원")
        print(f"  탄소격리 {r['탄소·배출 (룰베이스)']['탄소격리(tC/tBC)']} tC/tBC | 배출등급 {r['탄소·배출 (룰베이스)']['배출등급']}")
        if r.get('신뢰도 경고'):
            print(f"  ⚠ {r['신뢰도 경고']}")
