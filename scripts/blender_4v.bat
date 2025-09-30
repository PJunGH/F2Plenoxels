set gpu_id=0


for %%s in (chair hotdog lego mic ship drums materials) ^
do (

    python .\opt\opt_fewshot.py ^
        .\data\nerf_synthetic\%%s ^
        --gpu_id %gpu_id% ^
        -t .\logs\blender_4v\%%s ^
        -c .\opt\configs\syn_4v.json ^
        --hardcode_train_views 2 26 55 86

    python .\opt\render_imgs.py ^
        .\logs\blender_4v\%%s ^
        .\data\nerf_synthetic\%%s ^
        --gpu_id 0 ^
        --white_bkgd 1 ^
        --background_brightness 1.0
)



for %%s in (ficus) ^
do (

    python .\opt\opt_fewshot.py ^
        .\data\nerf_synthetic\%%s ^
        --gpu_id %gpu_id% ^
        -t .\logs\blender_4v\%%s ^
        -c .\opt\configs\syn_4v.json ^
        --hardcode_train_views 33 80 81 84

    python .\opt\render_imgs.py ^
        .\logs\blender_4v\%%s ^
        .\data\nerf_synthetic\%%s ^
        --gpu_id 0 ^
        --white_bkgd 1 ^
        --background_brightness 1.0
)
