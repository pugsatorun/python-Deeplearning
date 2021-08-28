import tensorflow as tf

# Mnistデータセットをロード
mnist = tf.keras.datasets.mnist

# データをロードしサンプルを整数から浮動小数点に変換
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train / 255.0, x_test / 255.0

# tf.keras.Sequentialモデルを構築　オプティマイザと損失関数を選ぶ。
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# クラスごとにロジットや対数オッズ比のスコアを算出する
predictions = model(x_train[:1]).numpy()
print(predictions)

# tf.nnsoftmax関数はクラスごとにこれらのロジットを確率に変換
print(tf.nn.softmax(predictions).numpy())

# losses.SparseCategoricalCrossentropyはロジットとTrueのインデックスに関するベクトルを入力にとり、それぞれの標本について損失スカラーを返す
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# model.fitは損失を最小化するようにモデルのパラメータを調整する。
model.fit(x_train, y_train, epochs=5)

# model.evaluateはモデルの性能を検査する
model.evaluate(x_test,  y_test, verbose=2)

# モデルが確率を返すようにしたい場合はモデルをラップしてソフトマックス関数を適用する
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))