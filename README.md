# Key frame extraction based on global motion statistics for team-sport videos
In this study, a global motion statistics-based key frame extraction scheme (KEGMS) [1] is proposed. The introduction of global motion results in fine-grained video partition and extracts the effective features for key frame extraction. A dataset called SportKF is built, which includes 25 videos of 197,878 frames in 112 min and 764 key frames from four types of sports (basketball, football, American football and field hockey). The experimental results demonstrate that the proposed scheme achieves state-of-the-art performance by introducing global motion statistics.
<br />
Considering the trade-off between speed and performance, we adopt the off-the-shelf PWC-Net [2] to estimate optical flow herein. This part of the code refers to a personal reimplementation [3] of PWC-Net using PyTorch. Should you be making use of this work, please cite PWC-Net accordingly. Also, make sure to adhere to the <a href="https://github.com/NVlabs/PWC-Net#license">licensing terms</a> of the authors.

Here is the poster I used when participating in ChinaMM 2020, after a little modification:

<a href="https://doi.org/10.1007/s00530-021-00777-7" rel="Paper"><img src="poster_ChinaMM2020.png?raw=true" alt="Paper" width="100%"></a>

For the related patents of this work, please see:
<br />
<a href="https://patents.google.com/patent/CN113032631A/en?oq=CN113032631A">Team sports video key frame extraction method based on global motion statistical characteristics</a>
<br />
<a href="https://patents.google.com/patent/CN113033308A/en?oq=CN113033308A">Team sports video game lens extraction method based on color features</a> (In the paper, I used "shot" as a video sequence photographed continuously by one camera, so the "lens" showed in the title of this patent may have been mistranslated.)

## Background
Key frame extraction is an important manner of video summarization. It can be used to interpret video content quickly. Existing approaches first partition the entire video into video clips by shot boundary detection, and then, extract key frames by frame clustering. However, in most team-sport videos, a video clip usually includes many events, and it is difficult to extract the key frames related to all of these events accurately, because different events of a game shot can have features of similar appearance. As is well known, most events in team-sport videos are attack and defense conversions, which are related to global translation. Therefore, by using fine-grained partition based on the global motion, a shot could be further partitioned into more video clips, from which more key frames could be extracted and they are related to the events.

In this study, global horizontal motion is introduced to further partition video clips into fine-grained video clips. Furthermore, global motion statistics are utilized to extract candidate key frames. Finally, the representative key frames are extracted based on the spatial–temporal consistence and hierarchical clustering, and the redundant frames are removed.

## Video summary extraction system (demo)
https://user-images.githubusercontent.com/112043923/205419245-7f6cc5e1-1f20-4d95-bb21-45a719ce6a70.mp4

## License
As stated in the <a href="https://github.com/NVlabs/PWC-Net#license">licensing terms</a> of the authors of the paper, the models are free for non-commercial share-alike purpose. Please make sure to further consult their licensing terms.

## References
```
[1]  @article{yuan2022key,
         title={Key frame extraction based on global motion statistics for team-sport videos},
         author={Yuan, Yuan and Lu, Zhe and Yang, Zhou and Jian, Meng and Wu, Lifang and Li, Zeyu and Liu, Xu},
         journal={Multimedia Systems},
         volume={28},
         number={2},
         pages={387--401},
         year={2022},
         publisher={Springer}
     }
```

```
[2]  @inproceedings{Sun_CVPR_2018,
         author = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
         title = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[3]  @misc{pytorch-pwc,
         author = {Simon Niklaus},
         title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
    }
```
