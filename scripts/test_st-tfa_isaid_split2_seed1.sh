python tools/detection/test.py configs/detection/st_tfa/isaid/split2/seed1/st-tfa/st-tfa_maskrcnn_r50_isaid-split2_seed1_10shot-fine-tuning.py work_dirs/st-tfa_maskrcnn_r50_isaid-split2_seed1_10shot-fine-tuning/iter_4000.pth --eval='bbox'
python tools/detection/test.py configs/detection/st_tfa/isaid/split2/seed1/st-tfa/st-tfa_maskrcnn_r50_isaid-split2_seed1_50shot-fine-tuning.py work_dirs/st-tfa_maskrcnn_r50_isaid-split2_seed1_50shot-fine-tuning/iter_7000.pth --eval='bbox'
python tools/detection/test.py configs/detection/st_tfa/isaid/split2/seed1/st-tfa/st-tfa_maskrcnn_r50_isaid-split2_seed1_100shot-fine-tuning.py work_dirs/st-tfa_maskrcnn_r50_isaid-split2_seed1_100shot-fine-tuning/iter_8000.pth --eval='bbox'
