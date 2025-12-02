
def generate_text_prompts(categories):
    caption = "A fundus photograph of [CLS]"
    prompts_dict = {"no diabetic retinopathy": ["no diabetic retinopathy", "no microaneurysms"],
                    "mild diabetic retinopathy": ["only few microaneurysms"],
                    "moderate diabetic retinopathy": ["many exudates near the macula", "hard exudates",
                                                      "many haemorrhages near the macula", "few severe haemorrhages",
                                                      "retinal thickening near the macula", "cotton wool spots"],
                    "severe diabetic retinopathy": ["venous beading", "many severe haemorrhages",
                                                    "intraretinal microvascular abnormality"],
                    "proliferative diabetic retinopathy": ["preretinal or vitreous haemorrhage", "neovascularization"],
                    "no retinal lesion": ["no retinal lesion"],
                    "tessellated fundus": ["tessellated fundus"],
                    "diffuse chorioretinal atrophy": ["diffuse chorioretinal atrophy"],
                    "patchy chorioretinal atrophy": ["patchy chorioretinal atrophy"],
                    "macular atrophy": ["macular atrophy"],
                    "noHR": ['no presence of hypertensive retinopathy', 'normal', 'no findings', 'normal optic disk'],
                    "HR": ['possible signs of haemorraghe with blot, dot, or flame-shaped',
                           'possible presence of microaneurysm, cotton-wool spot, or hard exudate',
                           'arteriolar narrowing', 'vascular wall changes', 'optic disk edema'],
                    "NG": ["no glaucoma"],
                    "G":  ["optic nerve abnormalities", "abnormal size of the optic cup",
                           "anomalous size in the optic disc"],
                    "normal": ["healthy", "no findings", "no lesion signs", "no glaucoma", "no retinopathy"],
                    "age related macular degeneration": ["many small drusen", "few medium-sized drusen", "large drusen",
                                                         "macular degeneration"],
                    "diabetic retinopathy": ["diabetic retinopathy"],
                    "glaucoma": ["optic nerve abnormalities", "abnormal size of the optic cup",
                                 "anomalous size in the optic disc"],
                    }

    prompts = {}
    for iCategory in categories:
        prompts[iCategory] = [caption.replace("[CLS]", iDescription) for iDescription in prompts_dict[iCategory]]
    return prompts