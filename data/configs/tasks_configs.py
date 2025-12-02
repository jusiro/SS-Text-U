
def get_task_cfg(task_name):

    if task_name == "SICAPv2":
        splits = {"train": "SICAPv2_train",   "test": "SICAPv2_test"}
    elif task_name == "Skin":
        splits = {"train": "Skin_train", "test": "Skin_test"}
    elif task_name == "NCT":
        splits = {"train": "NCT_train", "test": "NCT_test"}
    elif task_name == "CCRCC":
        splits = {"train": "CCRCC_train", "test": "CCRCC_test"}
    elif task_name == "MESSIDOR":
        splits = {"train": "MESSIDOR_train", "test": "MESSIDOR_test"}
    elif task_name == "MMAC":
        splits = {"train": "MMAC_A_train", "test": "MMAC_A_test"}
    elif task_name == "FIVES":
        splits = {"train": "FIVES_train", "test": "FIVES_test"}
    elif task_name == "mBRSET":
        splits = {"train": "mBRSET_train", "test": "mBRSET_test"}
    elif task_name == "CheXpert":
        splits = {"train": "CheXpert_train", "test": "CheXpert_test"}
    elif task_name == "NIH":
        splits = {"train": "nihlt_train", "test": "nihlt_test"}
    elif task_name == "COVID":
        splits = {"train": "covid_train", "test": "covid_test"}
    elif task_name == "PadChest":
        splits = {"train": "PadChest_train", "test": "PadChest_test"}
    else:
        print("Task not implemented...")
        return None

    return splits


def get_split_cfg(split_name) -> object:

    # Histology experiments
    if split_name == "SICAPv2_train":
        cfg = {"targets": ["NC", "G3", "G4", "G5"], 
               "dataframe": "histology/SICAPv2_train.csv", 
               "base_samples_path": "Histology/SICAPv2/images/",
               "modality": "histology"
               }
    elif split_name == "SICAPv2_test":
        cfg = {"targets": ["NC", "G3", "G4", "G5"],
                   "dataframe": "histology/SICAPv2_test.csv",
                   "base_samples_path": "Histology/SICAPv2/images/",
                   "modality": "histology"
                   }
    elif split_name == "NCT_train":
        cfg = {"targets": ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
                   "Normal colon mucosa", "Cancer-associated stroma",
                   "Colorectal adenocarcinoma epithelium"],
                   "dataframe": "histology/NCTCRC_train.csv",
                   "base_samples_path": "Histology/NCT-CRC/",
                   "modality": "histology"
                   }
    elif split_name == "NCT_test":
        cfg = {"targets": ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle",
                   "Normal colon mucosa", "Cancer-associated stroma",
                   "Colorectal adenocarcinoma epithelium"],
                   "dataframe": "histology/NCTCRC_test.csv",
                   "base_samples_path": "Histology/NCT-CRC/",
                   "modality": "histology"
                   }
    elif split_name == "Skin_train":
        cfg = {"targets": ['Necrosis', 'Skeletal muscle', 'Eccrine sweat glands', 'Vessels', 'Elastosis', 
                           'Chondral tissue', 'Hair follicle', 'Epidermis', 'Nerves', 'Subcutis', 'Dermis',
                           'Sebaceous glands', 'Squamous-cell carcinoma', 'Melanoma in-situ', 'Basal-cell carcinoma',
                           'Naevus'],
                "dataframe": "histology/Skin_train.csv",
                "base_samples_path": "Histology/Skin/",
                "modality": "histology"
                }
    elif split_name == "Skin_test":
        cfg = {"targets": ['Necrosis', 'Skeletal muscle', 'Eccrine sweat glands', 'Vessels', 'Elastosis',
                               'Chondral tissue', 'Hair follicle', 'Epidermis', 'Nerves', 'Subcutis', 'Dermis',
                               'Sebaceous glands', 'Squamous-cell carcinoma', 'Melanoma in-situ', 'Basal-cell carcinoma',
                               'Naevus'],
                   "dataframe": "histology/Skin_test.csv",
                   "base_samples_path": "Histology/Skin/",
                   "modality": "histology"
                   }
    elif split_name == "CCRCC_train":
        cfg = {"targets": ["blood", "cancer", "normal tissue", "stroma"],
                   "dataframe": "histology/CCRCC_train.csv",
                   "base_samples_path": "Histology/CCRCC/",
                   "modality": "histology"
                   }
    elif split_name == "CCRCC_test":
        cfg = {"targets": ["blood", "cancer", "normal tissue", "stroma"],
                   "dataframe": "histology/CCRCC_test.csv",
                   "base_samples_path": "Histology/CCRCC/",
                   "modality": "histology"
                   }

    # Ophthalmology experiments
    elif split_name == "MESSIDOR_train":
        cfg = {"targets": ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                               "severe diabetic retinopathy", "proliferative diabetic retinopathy"],
                   "dataframe": "fundus/MESSIDOR_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/02_MESSIDOR/",
                   "modality": "fundus"
                   }
    elif split_name == "MESSIDOR_test":
        cfg = {"targets": ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                               "severe diabetic retinopathy", "proliferative diabetic retinopathy"],
                   "dataframe": "fundus/MESSIDOR_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/02_MESSIDOR/",
                   "modality": "fundus"
                   }
    elif split_name == "MMAC_A_train":
        cfg = {"targets": ["no retinal lesion", "tessellated fundus", "diffuse chorioretinal atrophy",
                               "patchy chorioretinal atrophy", "macular atrophy"],
                   "dataframe": "fundus/MMAC_A_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/38_MMAC23/"
                                        "1.Classification/1.Images/1.Training/",
                   "modality": "fundus"
                   }
    elif split_name == "MMAC_A_test":
        cfg = {"targets": ["no retinal lesion", "tessellated fundus", "diffuse chorioretinal atrophy",
                               "patchy chorioretinal atrophy", "macular atrophy"],
                   "dataframe": "fundus/MMAC_A_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/38_MMAC23/"
                                        "1.Classification/1.Images/2.Validation/",
                   "modality": "fundus"
                   }
    elif split_name == "FIVES_train":
        cfg = {"targets": ['normal', 'diabetic retinopathy', 'glaucoma', 'age related macular degeneration'],
                   "dataframe": "fundus/FIVES_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/",
                   "modality": "fundus"
                   }
    elif split_name == "FIVES_test":
        cfg = {"targets": ['normal', 'diabetic retinopathy', 'glaucoma', 'age related macular degeneration'],
                   "dataframe": "fundus/FIVES_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/",
                   "modality": "fundus"
                   }
    elif split_name == "mBRSET_train":
        cfg = {"targets": ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                               "severe diabetic retinopathy", "proliferative diabetic retinopathy"],
                   "dataframe": "fundus/mBRSET_train.csv",
                   "base_samples_path": "Ophthalmology/CFP/51_mBRSET/images/",
                   "modality": "fundus"
                   }
    elif split_name == "mBRSET_test":
        cfg = {"targets": ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                               "severe diabetic retinopathy", "proliferative diabetic retinopathy"],
                   "dataframe": "fundus/mBRSET_test.csv",
                   "base_samples_path": "Ophthalmology/CFP/51_mBRSET/images/",
                   "modality": "fundus"
                   }

    # Radiology experiments
    elif split_name == "CheXpert_train":
        cfg = {"targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "cxr/CheXpert5x200_train.csv",
                   "base_samples_path": "CXR/CheXpert/CheXpert-v1.0/",
                   "modality": "cxr"
        }
    elif split_name == "CheXpert_test":
        cfg = {"targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "cxr/CheXpert5x200_test.csv",
                   "base_samples_path": "CXR/CheXpert/CheXpert-v1.0/",
                   "modality": "cxr"
        }
    elif split_name == "nihlt_train":
        cfg = {"targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                               "Nodule", "Pneumonia", "No Finding", "Pneumothorax", "Consolidation",
                               "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum",
                               "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "cxr/nih_train.csv",
                   "base_samples_path": "CXR/NIH/",
                   "modality": "cxr"
                   }
    elif split_name == "nihlt_test":
        cfg = {"targets": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                               "Nodule", "Pneumonia", "No Finding", "Pneumothorax", "Consolidation",
                               "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Pneumoperitoneum",
                               "Pneumomediastinum", "Subcutaneous Emphysema", "Tortuous Aorta",
                               "Calcification of the Aorta"],
                   "dataframe": "cxr/nih_test.csv",
                   "base_samples_path": "CXR/NIH/",
                   "modality": "cxr"
                   }
    elif split_name == "covid_train":
        cfg = {"targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                   "dataframe": "cxr/covid_train.csv",
                   "base_samples_path": "CXR/COVID-19_Radiography_Dataset/",
                   "modality": "cxr"
                   }
    elif split_name == "covid_test":
        cfg = {"targets": ["Normal", "COVID", "Pneumonia", "Lung Opacity"],
                   "dataframe": "cxr/covid_test.csv",
                   "base_samples_path": "CXR/COVID-19_Radiography_Dataset/",
                   "modality": "cxr"
                   }
    elif split_name == "PadChest_train":
        cfg = {"targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "cxr/PadChest_train.csv",
                   "base_samples_path": "CXR/",
                   "modality": "cxr"
        }
    elif split_name == "PadChest_test":
        cfg = {"targets": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
                   "dataframe": "cxr/PadChest_test.csv",
                   "base_samples_path": "CXR/",
                   "modality": "cxr"
        }
    else:
        print("Split not prepared...")
        return None

    return cfg
