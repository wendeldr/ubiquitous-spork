from sklearn import datasets
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.linear_model import Lasso


tf.config.list_physical_devices('GPU') 


df = datasets.load_breast_cancer()
df.data.shape


df['data'].shape


METRICS = [
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
    keras.metrics.MeanSquaredError(name='Brier score'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def early_stopping():
 return tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=5,
    mode='min',
    restore_best_weights=True)




data = np.concatenate([df['data'], df['target'].reshape(-1, 1)], axis=1)


# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(data, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df[:,-1]).reshape(-1, 1)
bool_train_labels = train_labels[:, 0] != 0
val_labels = np.array(val_df[:,-1]).reshape(-1, 1)
test_labels = np.array(test_df[:,-1]).reshape(-1, 1)

train_features = np.array(train_df[:,:-1])
val_features = np.array(val_df[:,:-1])
test_features = np.array(test_df[:,:-1])


# output_bias = keras.initializers.Constant(initial_bias)
N_value = train_features.shape[1]
inputs = keras.Input(shape=(N_value,))

# Reshape to (batch, N, 1) to treat each feature as a token.
x = keras.layers.Reshape((N_value, 1))(inputs)

# Self-attention block:
# Using one head with key_dim=1 so that the attention scores have shape (batch, 1, N, N)
attn_layer = keras.layers.MultiHeadAttention(num_heads=1,key_dim=8)
# We set return_attention_scores=True to obtain the attention matrix.
output_tensor, attn_scores = attn_layer(x, x, return_attention_scores=True, )
# Remove the head dimension so attn_scores becomes (batch, N, N)
# attn_scores = keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1))(attn_scores)

# Apply layer normalization to the attention matrix.
x = keras.layers.LayerNormalization()(output_tensor)

# Flatten the sequence dimension to make it compatible with dense layers
x = keras.layers.Flatten()(x)

# MLP block:
# Dense layers with sizes: 4*N, 2*N, N and output layer with 2 units.
x = keras.layers.Dense(4 * N_value, activation='relu')(x)
x = keras.layers.Dense(2 * N_value, activation='relu')(x)
x = keras.layers.Dense(N_value, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=METRICS
)


EPOCHS = 200
BATCH_SIZE = 32

history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels),
    verbose=1,
    callbacks=[early_stopping()])


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


mean_attn_array = []
attn = model.layers[2]

for x in train_features:
    _, attention_scores = attn(x.reshape(1,-1,1), x.reshape(1,-1,1), return_attention_scores=True) # take one sample
    mean_attn_array.append(attention_scores[0, 0, :, :])

mean_attn_array = np.array(mean_attn_array)
mean_attn = mean_attn_array.mean(axis=0)



plt.figure(figsize=(10,10))
sb.heatmap(mean_attn, annot=False, cbar=True,square=True, fmt='.2f')
plt.show()



mean_attn.sum(axis=0)


cols = df['feature_names']

# sort cols by max attention
att_sum = mean_attn.sum(axis=0)
cols[att_sum.argsort()[::-1]]


# Lasso Regression Analysis
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_features)

# Fit Lasso regression
lasso = Lasso(alpha=0.01)  # Small alpha for less aggressive regularization
lasso.fit(X_scaled, train_labels.ravel())

# Get feature importance (absolute coefficients)
lasso_importance = np.abs(lasso.coef_)

# Sort features by importance
lasso_feature_importance = pd.DataFrame({
    'feature': cols,
    'importance': lasso_importance
}).sort_values('importance', ascending=False)

print("\nLasso Regression Feature Importance:")
print(lasso_feature_importance)

# Compare with attention-based importance
att_sum = mean_attn.sum(axis=0)
attention_feature_importance = pd.DataFrame({
    'feature': cols,
    'importance': att_sum
}).sort_values('importance', ascending=False)

print("\nAttention-based Feature Importance:")
print(attention_feature_importance)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(lasso_importance)), lasso_importance)
plt.title('Lasso Regression Feature Importance')
plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.bar(range(len(att_sum)), att_sum)
plt.title('Attention-based Feature Importance')
plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
plt.tight_layout()
plt.show()





