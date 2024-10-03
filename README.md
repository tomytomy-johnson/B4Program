# B4Program

本プログラムは、"ゼロから作るDeep Learning 4"の第6章プログラムをもとに、学部卒業研究用に改変したものです。

プログラム引用元：https://github.com/oreilly-japan/deep-learning-from-scratch-4

## 概要
強化学習は，ある環境下から得られる報酬の総和が最大となるような行動をエージェントが試行錯誤をして学習する方法である．学習手法には，Q 学習やSarsa がよく用いられる．

Cliff Walking 問題は，強化学習で扱われる問題の一つで，崖付近を通る際の最適経路を学習する問題である．報酬は、崖に-100、崖以外のマスに-1が設定されており、より短いルートほど総報酬が高くなる。

この問題では、Q 学習が最短経路(optimal path)を学習し，Sarsa が迂回をする安全な経路(safe path)を学習する一方で，総報酬は崖の近くを通る Q学習よりも Sarsa の方が高いことが知られている．

本研究は、Broekens ら [1] が提唱した TDRL(Temporal DifferenceReinforcement Learning) 情動理論による将来の学習軌道に対する情動値を Q値に加えることを考案した．
これを行動選択の前に行うことで，より先の状態を見据えた行動選択を行うようにし，Q学習・Sarsa法の学習特徴を変容させ, より総報酬が高くすることを目的とする。

[1]Broekens, J., Chetouani, M., Towards Transparent Robot Learning Through TDRL-Based Emotional Expressions. IEEE Transactions on Affective Computing, vol.12, no.2, pp.352-362. 2021.

![Cliff-Walking](https://github.com/tomytomy-johnson/B4Program/blob/main/ch06/graph/CliffWalking-1.png)

## アルゴリズム
![algorithm](https://github.com/tomytomy-johnson/B4Program/blob/main/algorithm.png)
