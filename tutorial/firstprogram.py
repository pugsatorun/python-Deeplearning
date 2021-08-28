import tensorflow as tf

# Mnistデータセットをロード
mnist = tf.keras.datasets.mnist

# データをロードしサンプルを整数から浮動小数点に変換
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train / 255.0, x_test / 255.0


