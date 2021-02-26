import numpy as np
import os, os.path

list_videos   = np.load('data/processed/DictaSign/list_videos.npy')
length_videos = np.load('data/processed/DictaSign/length_videos.npy')
nVideos = list_videos.size

available_categories     = ['DS', 'FBUOY', 'PT', 'N', 'FS', 'G', 'ID']
available_categories_mod = ['DS', 'FBUOY', 'PT', 'N', 'FS', 'G', 'fls']
nCategories = len(available_categories)


annot_csv = np.genfromtxt('Dicta-Sign-LSF_Annotation.csv',delimiter=',',dtype=None)
annot_csv_video = annot_csv[1:, 0]
annot_csv_start = annot_csv[1:, 3].astype(int)
annot_csv_end   = annot_csv[1:, 4].astype(int)
annot_csv_cat   = annot_csv[1:, 5]
annot_csv_value = annot_csv[1:, 6]
nSegments = annot_csv_video.size

# creation d'une liste par type d'annotation
framewise_annotation = {}
for c in available_categories_mod:
    framewise_annotation['dataBrut_'+c] = []

# initialisation de tableaux à 0, pour chaque video dans chaque type de categorie
for c in available_categories_mod:
    for iV in range(nVideos):
        vidName = list_videos[iV]
        nFrames = int(length_videos[iV])#len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        framewise_annotation['dataBrut_'+c].append(np.zeros(nFrames))

# lecture de chaque segment du csv, et assignation des valeurs d'annotation dans le tableau
for iS in range(nSegments):
    videoS = annot_csv_video[iS][14:-10]
    startS = annot_csv_start[iS]
    endS   = annot_csv_end[iS]
    catS   = annot_csv_cat[iS]
    valueS = annot_csv_value[iS]

    indexVideoS = np.where(list_videos==videoS)[0]
    if indexVideoS.size > 0:
        indexVideoS = indexVideoS[0]
        if catS in available_categories:
            if catS == 'ID':
                framewise_annotation['dataBrut_fls'][indexVideoS][startS:endS+1] = int(valueS)
            else:
                framewise_annotation['dataBrut_'+catS][indexVideoS][startS:endS+1] = 1

np.savez('data/processed/DictaSign/annotations.npz', dataBrut_fls=framewise_annotation['dataBrut_fls'],
                                                     dataBrut_DS=framewise_annotation['dataBrut_DS'],
                                                     dataBrut_FBUOY=framewise_annotation['dataBrut_FBUOY'],
                                                     dataBrut_PT=framewise_annotation['dataBrut_PT'],
                                                     dataBrut_N=framewise_annotation['dataBrut_N'],
                                                     dataBrut_FS=framewise_annotation['dataBrut_FS'],
                                                     dataBrut_G=framewise_annotation['dataBrut_G'])
