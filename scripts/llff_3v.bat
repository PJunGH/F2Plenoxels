set gpu_id=0
::set decay_ratio=0.4

setlocal enabledelayedexpansion

for %%s in (fern flower fortress horns leaves orchids trex room) ^
do (
    
    if "%%s"=="fern" (
        set img_id1=1
        set img_id2=10
        set img_id3=19
    )^
    else if "%%s"=="flower" (
        set img_id1=1
        set img_id2=17
        set img_id3=33
    )^
    else if "%%s"=="fortress" (
        set img_id1=1
        set img_id2=21
        set img_id3=41
    )^
    else if "%%s"=="horns" (
        set img_id1=1
        set img_id2=30
        set img_id3=61
    )^
    else if "%%s"=="leaves" (
        set img_id1=1
        set img_id2=12
        set img_id3=25
    )^
    else if "%%s"=="orchids" (
        set img_id1=1
        set img_id2=12
        set img_id3=23
    )^
    else if "%%s"=="room" (
        set img_id1=1
        set img_id2=20
        set img_id3=39
    )^
    else if "%%s"=="trex" (
        set img_id1=1
        set img_id2=28
        set img_id3=54
    )

    echo !img_id2!

    python .\opt\opt_fewshot.py ^
        .\data\nerf_llff_data\%%s ^
        --gpu_id %gpu_id% ^
        -t .\logs\llff_3v\%%s ^
        -c .\opt\configs\llff_3v.json ^
        --hardcode_train_views !img_id1! !img_id2! !img_id3!

    python .\opt\render_imgs.py ^
        .\logs\llff_3v\%%s ^
        .\data\nerf_llff_data\%%s ^
        --gpu_id 0 ^
        --background_brightness 0.5
)