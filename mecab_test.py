#%%
import MeCab
m = MeCab.Tagger()
out = m.parse('마캅이 잘 설치되었는지 확인중입니다.')
print(out)

#%%
