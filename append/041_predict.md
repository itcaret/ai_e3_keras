# 4 学習済みモデルの推論

機械学習ではモデルの学習時に多くのコンピュータリソースを使用します。たとえばCNNのような計算量の多い学習になるとGPUを搭載したマシンを用意する必要があるでしょう。

<img src="img/11_train.png?aa" width="480px">

それでは学習済みのモデルを活用する場合はどうでしょうか。学習済みのモデルに入力を与えて出力を得ることを本書では推論と呼びますが、推論は学習時に比べて少量のコンピュータリソースで処理できます。

<img src="img/11_predict.png?aa" width="480px">


ここからは学習済みのモデルを活用する方法を具体的に見ていきましょう。

<div style="page-break-before:always"></div>

## 4.1 学習済みモデルの準備

ここでは事前準備としてMNISTデータを学習したモデルを作成します。

```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="adagrad",
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=64, epochs=10)

model.save('my_trained_model.h5')
```

上記のプログラムを実行すると学習済みのモデルデータmy_trained_model.h5が生成されます。以降はこの学習済みモデルを活用する方法を取り上げます。

<div style="page-break-before:always"></div>

また以下の画像データがカレントフォルダ上にあるものとします。

+ mnist0.jpg・・・0の手書き画像
+ mnist1.jpg・・・1の手書き画像
+ mnist2.jpg・・・2の手書き画像
+ mnist3.jpg・・・3の手書き画像
+ mnist4.jpg・・・4の手書き画像
+ mnist5.jpg・・・5の手書き画像
+ mnist6.jpg・・・6の手書き画像
+ mnist7.jpg・・・7の手書き画像
+ mnist8.jpg・・・8の手書き画像
+ mnist9.jpg・・・9の手書き画像

<img src="img/11_my_mnist.png?aa">


<div style="page-break-before:always"></div>

## 4.2 学習済みモデルの推論

学習済みのモデルを使って入力データを推論（分類）してみましょう。学習済みモデルによって推論を行うにはpredictメソッドを使います。

```python
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

files = ["mnist0.jpg", "mnist1.jpg", "mnist2.jpg", "mnist3.jpg", "mnist4.jpg"]
images = []
for file in files:
    img = img_to_array(load_img(file, target_size=(28, 28), grayscale=True))
    img = img.astype("float32")
    img /= 255
    images.append(img)
images = np.array(images)

model = load_model('my_trained_model.h5')
result = model.predict(images)

print(result.shape)
for i in range(result.shape[0]):
    print(result[i])
    print(np.argmax(result[i]))
```

プログラムを実行すると次のような結果が表示されるでしょう。

```
(5, 10)
[  9.99895453e-01   8.10308354e-09   7.54248977e-05   1.16817023e-06
   7.36907140e-08   2.16216307e-07   2.16065037e-05   1.42526162e-07
   2.66369057e-06   3.16157616e-06]
0
[  9.01548333e-07   9.97682452e-01   4.24800965e-05   1.42138206e-05
   1.04314684e-04   4.19998469e-05   1.31453471e-05   1.26454397e-04
   1.96028897e-03   1.36981107e-05]
1
[  2.74680584e-04   6.82780568e-11   9.99039054e-01   4.95960528e-04
   1.32802313e-07   1.02595698e-06   2.09289812e-08   1.21227175e-04
   6.57433993e-05   2.21645814e-06]
2
[  4.37711788e-06   1.27913445e-05   8.46660478e-05   9.96545017e-01
   9.56662376e-08   2.62512453e-03   4.01181843e-08   9.97625520e-08
   8.03857329e-05   6.47358363e-04]
3
[  6.05275519e-09   8.81599681e-06   2.86308677e-05   3.33256509e-07
   9.99942899e-01   4.66654893e-09   2.33737822e-07   1.20231721e-06
   1.33067333e-05   4.54389783e-06]
4
```

MNISTのような分類問題のケースではpredictメソッドは戻り値にクラス（0〜9）に対する確率を返します。例えば1件目の結果を注目してみましょう。

```
[  9.99895453e-01   8.10308354e-09   7.54248977e-05   1.16817023e-06
   7.36907140e-08   2.16216307e-07   2.16065037e-05   1.42526162e-07
   2.66369057e-06   3.16157616e-06]
0
```

配列の先頭要素が99%と一番大きな値になっています。つまり0と予測していることになります。

<div style="page-break-before:always"></div>

プログラムの詳細を見てみましょう。まずは推論時に使う入力データをNumPy配列に追加しています。

```python
files = ["mnist0.jpg", "mnist1.jpg", "mnist2.jpg", "mnist3.jpg", "mnist4.jpg"]
images = []
for file in files:
    img = img_to_array(load_img(file, target_size=(28, 28), grayscale=True))
    img = img.astype("float32")
    img /= 255
    images.append(img)
images = np.array(images)
```

変数imagesには0〜4の手書きデータが保持されます。

> 変数imagesのshapeは（5,28,28）となります。

続いて保存していた学習済みモデルをロードします。

```python
model = load_model('my_trained_model.h5')
```

学習済みモデルをロードしたらpredictメソッドを使って推論を行います。

```python
result = model.predict(images)
```

predictメソッドは引数に複数件のサンプルを受け取ります。たとえばMNISTデータであれば1件のサンプルは(28,28)であるため、ここでは10件のサンプル(10,28,28)を引数に指定しています。

> グレースケールであるためチャネル数は1です。（10,28,28,1）としても動作します。

MNISTのような分類問題のケースではpredictメソッドは戻り値にクラスに対する確率を返します。

```
[ 0.  0.2  0.  0.1  0.  0.  0.  0.  0.7  0.]
```

上記の出力となった場合、推論した結果、1である確率が20%、4である確率が10%、8である確率が70%という具合です。

```
[ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
```

同様に上記の出力となった場合、モデルは3である確率が100%と推論しています。

<div style="page-break-before:always"></div>

### predict_classes

学習済みのモデルはpredict_classesメソッドによって分類結果を確率ではなく、クラスラベルで出力することもできます。

```python
result_classes = model.predict_classes(images, verbose=0)

print(result_classes.shape)
print(result_classes)
```

上記のプログラムを実行すると次のように出力されます。

```
(5,)
[0 1 2 3 4]
```

結果が確率ではなくクラスラベルで出力されているのがわかります。
