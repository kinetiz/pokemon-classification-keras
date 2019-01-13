import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- โหลดข้อมูลจาก csv
pkm = pd.read_csv("pokemon.csv")

# ส่องข้อมูล ดูว่าข้อมูลเป็นยังไง
pkm.head()
# ตรวจสอบคุณภาพข้อมูล เช่น missing data, null, duplicate
pkm.info()

# เราต้องการ ทำนาย ชนิดของ pokemon จาก stats ของ pokemon เพราะงั้นเราสร้างตัวแปล pkm_stats เพื่อใช้เป็น Feature ในการทำนาย
pkm_stats = pkm[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']]

# เนื่องจากข้อมูลเป็น True / False เราต้องแปลงเป็น 0 / 1 สำหรับการเทรนโมเดล
pkm_stats['Legendary'] = pd.get_dummies(pkm_stats['Legendary'], drop_first= True)

pkm_stats.head()

# เราต้องการทำนาย Type ของ pokemon แต่ข้อมูลดิบมาเป็น text อยู่ เราต้องแปลงเป็น label 0 หรือ 1 และ แยกจำนวน Column ตาม Type ของ pokemon
# ซึ่งเราสามารถใช้คำสั่ง pandas.get_dummies เพื่อแปลง Categorical data เป็น Label ได้ ส่วน drop_first หมายถึง เรา drop column แรกที่เกินมา
# เช่น [True,False,True] จะถูกแปลงเป็น 2 columns อันแรกแทน True อันสองแทน False จากข้อมูลจะได้ [(1, 0),(0, 1),(1, 0)]
# จะเห็นว่าจริงๆ แค่ column แรกก็สามารถแยกแยะได้แล้ว เราจึง drop column ส่วนเกินอันนี้ซะเพื่อจะได้ไม่ทำให้โมเดลเราซับซ้อนโดยไม่จำเป็น
pkm_type = pd.get_dummies(pkm[['Type 1']], drop_first=True)
LABELS_TABLE = pd.DataFrame(pkm_type.columns.values)
pkm_type.info()

# ไว้ตรวจสอบว่าจำนวน row ของ feature กับ target ต้องเท่ากันนะ ไม่งั้นไม่ให้ไปต่อ
if pkm_stats.shape[0] == pkm_type.shape[0]:
    print("* * * Features and targets are compatible! :) * * *")
else:
    raise("Data size of Features and targets are not compatible...")

### ========= Prepare data ================
def train(X, Y):
    # สร้างโครง neural network
    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1], activation='softmax'))

    # สร้างเสร็จแล้ว compile แล้วเลือก Loss function, เทคนิคการ optimize, metric ที่ใช้วัดคุณภาพโมเดล
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # เทรนโมเดล fit หมายถึง สั่งโมเดลให้ฟิต label จาก data ที่ใส่ลงไป
    model.fit(X, Y, epochs=500, batch_size=800)

    return model

# แบ่งข้อมูลเป็น Training & Test set = 70/30 และ ทำให้แบ่งข้อมูลเหมือนเดิมทุกครั้งด้วยการ set random state
xtr, xts, ytr, yts = train_test_split(pkm_stats, pkm_type, test_size=0.3, random_state=1)

# เตรียมข้อมูลแบบ standardise แล้วไว้เทียบผลกันว่าแบบไหนดีกว่า
scaler = StandardScaler().fit(xtr.values)
sxtr = scaler.transform(xtr.values)
sxts = scaler.transform(xts.values)


# เตรียมข้อมูลในโครงสร้างที่ใช้สำหรับเทรน Neural Network ต้องแปลงเป็น numpy array ใช้ .values ง่ายๆ เลย
model1 = train(X=xtr.values, Y=ytr.values)
model2 = train(X=sxtr, Y=ytr.values)

print('===== non-standardised data =====')
# วัคความแม่นโมเดลด้วยข้อมูลที่ใช้เทรน หรือ Training data
print('Evaluate with Training data: {}'.format(model1.evaluate(xtr, ytr)))

# วัดผลด้วย Test data
print('Evaluate with Test data: {}'.format(model1.evaluate(xts, yts)))

print('===== Standardised data =====')
# วัคความแม่นโมเดลด้วยข้อมูลที่ใช้เทรน หรือ Training data
print('Evaluate with Training data: {}'.format(model2.evaluate(sxtr, ytr)))

# วัดผลด้วย Test data
print('Evaluate with Test data: {}'.format(model2.evaluate(sxts, yts)))



# # แปะ type ที่ทำนายไว้ไฟล์ csv เพื่อเทียบผล
# predicted_class = model.predict_classes(data)
# lab_predicted_class = [LABELS_TABLE.at[i,0][7:] for i in predicted_class ]
# pkm['Pred_Type1'] = lab_predicted_class
# pkm.to_csv('predicted_pokemon.csv')
