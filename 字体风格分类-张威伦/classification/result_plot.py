import utils as utils
from utils.draw import *
from torchvision import transforms
import os
from model import efficientnetv2_s
from model import ClassificationNet as create_model

if __name__ == '__main__':

    # create model
    embedding_net = efficientnetv2_s()
    model = create_model(embedding_net, n_classes=4).to("cpu")
    # load model weights
    model_weight_path = "./weights/model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
    model.eval()

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_images_path, train_images_label, val_images_path, val_images_label = utils.read_split_data("C:\project\category_classification\dataset")
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])}
    train_dataset = utils.MyDataSet(images_path=train_images_path,
                                    images_class=train_images_label,
                                    transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = utils.MyDataSet(images_path=val_images_path,
                                  images_class=val_images_label,
                                  transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_baseline, train_labels_baseline)
    val_embeddings_baseline, val_labels_baseline = extract_embeddings(val_loader, model)
    plot_embeddings(val_embeddings_baseline, val_labels_baseline)