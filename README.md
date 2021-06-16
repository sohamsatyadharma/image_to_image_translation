# Image-to-image-transalation

We worked on paired and unpaired image to image translation on multiple image domains, following the [Pix2pix](https://arxiv.org/pdf/1611.07004.pdf) and [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf) papers for the ECE 228 course project in Spring 2021. We conducted various experiments on the models to understand how they work and used qualitative and quantitative metrics to assess which model performs best. We used new metrics in [Precision and Recall](https://arxiv.org/pdf/1904.06991v1.pdf), that had not been previously used for the image to image translation task to the best of our knowledge. We used [this](https://github.com/manicman1999/precision-recall-keras) implementation of precision and recall for our project. We found that they broadly agreed with the FID score and the qualitative results. The details about the models, experiments performed and the results are provided in our project report.

To train the pix2pix model for paired image-to-image translation on facade dataset for 150 epochs and batch size=16, run -
python run.py ----train_type paired

To train the cycle GAN model for paired image-to-image translation on horse2zebra dataset for 150 epochs and batch size=8, run - 
python run.py ----train_type unpaired

To calculate Inception FID score, run - 
python evaluate_inception.py --fake_dir “../data/fake”   --real_dir “../data/real” 

To calculate Precision-Recall scores, run - 
python evaluate_precision_recall.py --fake_dir “../data/fake”   --real_dir “../data/real” --number_of_images 256
