import keras
import sys
import h5py
import numpy as np

model_filename = str(sys.argv[1])
data_filename = str(sys.argv[2])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    return x_data / 255, y_data


def valid_accu(model, x, y):
    clean_label = np.argmax(model.predict(x), axis=1)
    clean_accu = np.mean(np.equal(clean_label, y)) * 100
    print('Classification accuracy for ', model.name, ':', clean_accu, '%')
    return clean_accu


def do_repair():
    # load validation data
    x_valid, y_valid = data_loader(data_filename)
    # load badnet model
    bd_model = keras.models.load_model(model_filename)

    # prune neurons of layer 'conv_3'
    neuron_activate = keras.Model(inputs=bd_model.input,
                                  outputs=bd_model.get_layer("conv_3").output
                                  ).predict(x_valid)
    accu_ori = valid_accu(bd_model, x_valid, y_valid)
    accu_threshold = accu_ori * 0.95
    conv3_num = neuron_activate.shape[3]
    prune_sort = np.argsort(np.mean(neuron_activate, axis=(0, 1, 2)))
    layer_prune = bd_model.get_layer("conv_3")
    prune_count = 0
    for i in range(conv3_num):
        channel = prune_sort[i]
        ori_w = np.array(layer_prune.get_weights()[0])
        ori_b = np.array(layer_prune.get_weights()[1])
        weights_w = np.array(ori_w)
        weights_b = np.array(ori_b)
        weights_w[:, :, :, channel] = 0.
        weights_b[channel] = 0.
        layer_prune.set_weights([weights_w, weights_b])
        accu = valid_accu(bd_model, x_valid, y_valid)
        if accu < accu_threshold:
            layer_prune.set_weights([ori_w, ori_b])
            break
        prune_count += 1
    print("Pruning number of neurons: ", prune_count)
    print("bd_model layers:")
    for i in bd_model.layers:
        print(i.name)

    # retrain with validation data
    # only update the weights of the last layer
    N = bd_model.output.shape[1]
    tune_base_model = keras.Model(inputs=bd_model.input,
                                  outputs=bd_model.get_layer("output").input)
    defence_model = bd_model
    defence_model.compile(optimizer="Adam",
                          loss='categorical_crossentropy',
                          metrics=["accuracy"])
    for layer in tune_base_model.layers:
        layer.trainable = False
    print("defence_model layers with trainable state in first tuning:")
    for layer in defence_model.layers:
        print(layer.name, layer.trainable)
    y_valid_tune = np.zeros((len(y_valid), N))
    for i, y in enumerate(y_valid):
        y = int(y)
        y_valid_tune[i][y] = 1
    defence_model.fit(x_valid, y_valid_tune, epochs=5)

    # further fine-tuning
    # update the weights of the layers after 'conv_3'
    for layer in reversed(tune_base_model.layers):
        if layer.name == 'conv_3':
            break
        layer.trainable = True
    print("defence_model layers with trainable state in second tuning:")
    for layer in defence_model.layers:
        print(layer.name, layer.trainable)

    defence_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                          loss='categorical_crossentropy',
                          metrics=["accuracy"])
    for i in range(6):
        print("Round", i+1)
        defence_model.fit(x_valid, y_valid_tune, epochs=5)
        if valid_accu(defence_model, x_valid, y_valid) >= accu_ori:
            break

    defence_model.save(model_filename[:len(model_filename)-3]+"_defence.h5")


if __name__ == '__main__':
    do_repair()
