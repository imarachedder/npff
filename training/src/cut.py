import pandas as pd


cntbtrs_clnts_ops_trn = pd.read_csv('./test_data/cntrbtrs_clnts_ops_tst.csv', sep=';', encoding='cp1251')
# trnsctns_ops_trn = pd.read_csv('./test_data/trnsctns_ops_tst.csv',sep=';', encoding='cp1251')


cntbtrs_clnts_ops_trn = cntbtrs_clnts_ops_trn.head(200)
# trnsctns_ops_trn = trnsctns_ops_trn.head(5)
cntbtrs_clnts_ops_trn.to_csv("./out/cntrbtrs_clnts_ops_trn.csv", sep=';',)
# trnsctns_ops_trn.to_csv("./out/trnsctns_ops_trn.csv", sep=';')
