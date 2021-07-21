# The final code for OGB
# Installation Requirements:
1. ogb=1.3.1
2. torch=1.7.0
3. torch-geometric=1.7.0
4. torch-scatter=2.0.6
5. torch-sparse=0.6.9

# Baseline models
1. The path of ogbg-ppa dataset should be ogb/ppa/dataset/ogbg_ppa, then you can get the performance reported by running the default code 10 times. In fact, we can get 82 by other parameters, but the model will be too large and training is much more slower. So in this report, we only set the parameters for the performance 81.4.
2. The path of ogbg-code2 dataset should be ogb/code/dataset/ogbg_code2.
Then you can get the performance reported by running the default code 10 times
3. The path of ogbg-molhiv dataset should be ogb/molhiv/dataset/ogbg_molhiv. We run 10 different random forest models for FingerPrint, then join train a deep gnn model for each random forest model. You can get the 10 results by running the default code as follows.
4. The path of ogbg-molpcba dataset should be obg/molpcba/dataset/ogbg_molpcba. Then you can get the performance reported by running the default code 10 times.

## ogbg-ppa
1. cd ogb/ppa; 
2. sh run_script.sh 0

## ogbg-code2
1. cd ogb/code;
2. sh run_script.sh 0

## ogbg-molhiv
We follow the [Neural FingerPrints](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv) to get MorganFingerprint and MACCSFingerprint. Instead of adopting the proposed Soft-MorganFingerprint, we find that these two conventional fingerprints already serve as a strong baseline.

1. cd ogb/molhiv
2. python extract_fingerprint.py
3. python random_forest.py
4. sh run_scipt.sh 0

## ogbg-molpcba
1. cd ogb/molpcba
2. sh run_script.sh 0

# Performance
bot : bag of tricks
## ogbg-molhiv
| methods | test accuracy | val accuracy | Parameters | Hardware |
| ------ | ------ | ----- | ----- | ----- |
| GINE+bot | 0.2994±0.0019 | 0.3094±0.0023 | 5.26M | Tesla V100 32GB |

## ogbg-molpcba
| methods | test accuracy | val accuracy | Parameters | Hardware |
| ------ | ------ | ----- | ----- | ----- |
| FingerPrint+GMAN | 0.8244±0.0033 | 0.8329±0.0039 | 1.37M | Tesla V100 32GB |

## ogbg-ppa
| methods | test accuracy | val accuracy | Parameters | Hardware |
| ------ | ------ | ----- | ----- | ----- |
| ExpC+bot | 0.8140±0.0028 | 0.7811±0.0012 | 3.7M | Tesla V100 32GB |

## ogbg-code2
| methods | test accuracy | val accuracy | Parameters | Hardware |
| ------ | ------ | ----- | ----- | ----- |
| GMAN+bot | 0.1770±0.0012 | 0.1631±0.0090 | 63.7M | Tesla V100 32GB |

# Reference:
1. https://github.com/qslim/epcb-gnns
2. https://github.com/devnkong/FLAG
3. https://github.com/RBrossard/GINEPLUS
