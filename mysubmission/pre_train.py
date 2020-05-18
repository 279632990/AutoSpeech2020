import os
from data_manager import DataManager
from ingestion.dataset import AutoSpeechDataset
from resnet_utils.resnet_model import resnet34Model

def preTrainModel(net, data_path=None):
    D = AutoSpeechDataset(os.path.join("../sample_data/test_data1", 'data01.data'))
    D.read_dataset()
    metadata = D.get_metadata()
    data_manager = DataManager(metadata, D.get_train())
    train_x, train_y, val_x, val_y = data_manager.get_train_data(train_loop_num=11,
                                                                       model_num= 1,
                                                                       round_num= 2,
                                                                       use_new_train=True,
                                                                       use_mfcc=True)
    net.init_model(train_x.shape[1:], metadata["class_num"])
    net.fit(train_x, train_y, (val_x, val_y), 2)
    result = net.predict(val_x)
    print('a')
if __name__ == '__main__':
    preTrainModel(resnet34Model())