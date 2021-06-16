# Image-to-image-transalation

To train the pix2pix model for paired image-to-image translation on facade dataset for 150 epochs and batch size=16, run -
python run.py ----train_type paired

To train the cycle GAN model for paired image-to-image translation on horse2zebra dataset for 150 epochs and batch size=8, run - 
python run.py ----train_type unpaired

To calculate Inception FID score, run - 
python evaluate_inception.py --fake_dir “../data/fake”   --real_dir “../data/real” 

To calculate Precision-Recall scores, run - 
python evaluate_precision_recall.py --fake_dir “../data/fake”   --real_dir “../data/real” --number_of_images 256



