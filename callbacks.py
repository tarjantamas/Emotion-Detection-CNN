from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

modelCheckpoint = ModelCheckpoint('models/keras/test/weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc')
tensorBoard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                            write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
lrReducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)