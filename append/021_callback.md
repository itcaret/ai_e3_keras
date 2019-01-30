# 2 効率の良い学習の進め方

機械学習を効率よく進めるには次のような課題が出てきます。

+ 学習状況の出力
  + 各エポックの出力をコンソールではなくファイルに出力したい。
+ 定期的なモデルの保存
  + 各エポックの終了時点のモデルを保存したい。
+ 不要な学習の停止
  + 学習の進んでいない状況に陥った場合、学習を終了したい。

Kerasにはコールバックという仕組みが用意されています。コールバックを使えば上記の要件を達成することができます。本章では以下の3つのコールバックの利用方法について取り上げます。

+ CSVLogger
+ ModelCheckpoint
+ EarlyStopping

<div style="page-break-before:always"></div>

## 2.1 学習状況の出力（CSVLogger）

CSVLoggerは各エポックの結果をcsvファイルに保存するコールバックです。ここではMNISTデータを処理するケースに利用してみましょう。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import CSVLogger

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(100, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

csv_logger = CSVLogger('training.log')
history = model.fit(x_train, y_train, batch_size=64, epochs=5,
                                validation_data=(x_test, y_test),
                                callbacks=[csv_logger])
```

プログラムを実行すると次のように表示されるでしょう。

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 4s - loss: 0.8582 - acc: 0.7899 - val_loss: 0.4642 - val_acc: 0.8796
Epoch 2/5
60000/60000 4s - loss: 0.4241 - acc: 0.8862 - val_loss: 0.3610 - val_acc: 0.9025
Epoch 3/5
60000/60000 4s - loss: 0.3578 - acc: 0.9003 - val_loss: 0.3211 - val_acc: 0.9127
Epoch 4/5
60000/60000 4s - loss: 0.3242 - acc: 0.9092 - val_loss: 0.2953 - val_acc: 0.9206
Epoch 5/5
60000/60000 4s - loss: 0.3012 - acc: 0.9154 - val_loss: 0.2790 - val_acc: 0.9231
```

またカレントフォルダ上にtrain.logファイルが出力されていることを確認しておきましょう。

```
epoch,acc,loss,val_acc,val_loss
0,0.789916666667,0.858214442666,0.8796,0.464201840687
1,0.88625,0.424075638851,0.9025,0.361021899486
2,0.900333333333,0.357753000212,0.9127,0.321085527372
3,0.909233333333,0.324181920926,0.9206,0.295257022667
4,0.9154,0.301182911809,0.9231,0.278953828871
```

CSVLoggerによって各エポック終了時の出力がCSVファイルに出力されているのがわかります。

プログラムの詳細を見てみましょう。Kerasのコールバックパッケージ（keras.callbacks）からCSVLoggerをインポートします。

```python
from keras.callbacks import CSVLogger
```

CSVLoggerはコンストラクタの引数に出力ファイルパスを受け取ります。

```python
csv_logger = CSVLogger('training.log')
```

次のようにseparator（区切り文字）やappend（追記型かどうか）をオプション引数に指定することもできます。

```python
csv_logger = CSVLogger('training.log', separator=',', append=False)
```

生成したコールバックはモデルのfitメソッドのcallbacks引数に指定します。

```python
history = model.fit(x_train, y_train, batch_size=64, epochs=5,
                                validation_data=(x_test, y_test),
                                callbacks=[csv_logger])
```

> コールバックは複数指定可能であるため、引数を配列で渡している点に注意してください。

<div style="page-break-before:always"></div>

## 2.2 定期的なモデルの保存（ModelCheckpoint）

ModelCheckpointは各エポック終了後にモデルを保存します。


```python
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(100, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint('my_model.{epoch:02d}-{val_loss:.2f}.hdf5')
history = model.fit(x_train, y_train, batch_size=64, epochs=5,
                                validation_data=(x_test, y_test),
                                callbacks=[checkpointer])
```

プログラムを実行すると次のように表示されるでしょう。

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/5
60000/60000 3s - loss: 0.8895 - acc: 0.7792 - val_loss: 0.4654 - val_acc: 0.8809
Epoch 2/5
60000/60000 4s - loss: 0.4221 - acc: 0.8854 - val_loss: 0.3601 - val_acc: 0.9036
Epoch 3/5
60000/60000 4s - loss: 0.3552 - acc: 0.9009 - val_loss: 0.3199 - val_acc: 0.9116
Epoch 4/5
60000/60000 3s - loss: 0.3223 - acc: 0.9084 - val_loss: 0.2950 - val_acc: 0.9180
Epoch 5/5
60000/60000 4s - loss: 0.3003 - acc: 0.9149 - val_loss: 0.2805 - val_acc: 0.9212
```

またカレントフォルダ上に次のような5つのモデルデータが保存されている点を確認しておきましょう。

+ my_model.00-0.47.hdf5
+ my_model.01-0.36.hdf5
+ my_model.02-0.32.hdf5
+ my_model.03-0.29.hdf5
+ my_model.04-0.28.hdf5

各エポックごとの学習済みモデルが保存されているのがわかります。


プログラムの詳細を見てみましょう。Kerasのコールバックパッケージ（keras.callbacks）からModelCheckpointをインポートします。

```python
from keras.callbacks import CSVLogger
```

ModelCheckpointはコンストラクタの引数に出力ファイルパスを受け取ります。

```python
checkpointer = ModelCheckpoint('my_model.{epoch:02d}-{val_loss:.2f}.hdf5')
```

出力ファイルパスにはエポック数（epoch）と出力に含まれるメトリクス（val_loss、val_accなど）を指定することができます。上記の指定の場合、各エポックの終了時にエポック数、テストデータの損失値（val_loss）がファイル名に利用されます。


periodオプションを指定すればファイル出力するエポックの感覚を変更できます。

```python
checkpointer = ModelCheckpoint('my_model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                period=2)
```

上記の場合、2エポックごとに学習済みモデルが保存されます。

+ my_model.01-0.36.hdf5
+ my_model.03-0.29.hdf5

> periodオプションのデフォルトは1です。

またsave_weights_onlyオプションを指定すればモデルの重みだけ保存することもできます。

```python
checkpointer = ModelCheckpoint('my_model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                save_weights_only=True)
```

save_best_onlyオプションを指定すれば、学習の進んだ場合のみ学習済みモデルを保存することができます。monitorオプションで監視する値を指定します。

```python
checkpointer = ModelCheckpoint('my_model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                save_best_only=True, monitor="val_loss")
```

> monitorオプションのデフォルトはval_lossです。エポックごとの出力に含まれるメトリクス（acc, val_accなど）を指定できます。


<div style="page-break-before:always"></div>

## 2.3 不要な学習の停止（EarlyStopping）

EarlyStoppingは、監視する値の変化が停止した時に訓練を終了します。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# reduce data
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(100, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

earyl_stopping = EarlyStopping(verbose=1)
history = model.fit(x_train, y_train, batch_size=64, epochs=500,
                                validation_data=(x_test, y_test),
                                callbacks=[earyl_stopping])
```


> 上記のプログラムはKerasのAPIの挙動を確認するために訓練データ、検証データを100件にしています。

```
# reduce data
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]
```

プログラムを実行すると次のように表示されるでしょう。

```
・・・省略
100/100 0s - loss: 0.2474 - acc: 1.0000 - val_loss: 0.9438 - val_acc: 0.6700
Epoch 221/500
100/100 0s - loss: 0.2457 - acc: 1.0000 - val_loss: 0.9422 - val_acc: 0.6700
Epoch 222/500
100/100 0s - loss: 0.2440 - acc: 1.0000 - val_loss: 0.9415 - val_acc: 0.6700
Epoch 223/500
100/100 0s - loss: 0.2426 - acc: 1.0000 - val_loss: 0.9419 - val_acc: 0.6700
Epoch 00222: early stopping
```

EarlyStoppingはデフォルトでval_loss（検証データの損失値）を監視対象とします。実行結果を見ると500回のエポックの途中、222回目のエポックでval_lossが最小となっており、223回目のエポックではval_lossが増加しているのがわかります。このようなケースを検出するとEarlyStoppingは早期に学習を終了します。

> EarlyStopping(verbose=1) のようにverboseオプションを有効にすると出力ログに Epoch 00222: early stopping が含まれるようになります。


EarlyStoppingはmonitorオプションで監視する値を変更できます。デフォルトはval_lossです。

```python
earyl_stopping = EarlyStopping(monitor='val_acc')
```

上記のように指定するとval_acc（検証データの正答率）を監視対象にします。

min_deltaで最小変化値を指定することもできます。指定した値より変化が小さければ改善していないと判定します。デフォルトは0です。

```python
earyl_stopping = EarlyStopping(min_delta=0.01)
```

上記のように指定するとval_lossの改善が0.01より小さいときに学習を終了します。

patienceオプションで監視する値が改善しなくなってからのエポック数を指定することができます。デフォルトは0です。

```python
earyl_stopping = EarlyStopping(patience=3)
```

上記のように指定するとval_lossの改善が3回連続で発生しなかった場合に学習を終了します。

<div style="page-break-before:always"></div>

## 2.4 その他のコールバック

Kerasのコールバックについては以下のマニュアルページに整理されています。

https://keras.io/ja/callbacks/

LearningRateSchedulerやReduceLROnPlateauコールバックを使えば学習時に学習率を調整していくこともできます。またKerasの内部エンジンにTensorFlowを使っている場合はTensorBoardコールバックによって学習状況をTensorBoard上で確認することもできます。
