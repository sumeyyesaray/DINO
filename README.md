# DINO Frozen Backbone Feature Evaluation

Amaç: Önceden eğitilmiş, dondurulmuş DINOv2/DINOv3 backbone'unun çıkardığı
özelliklerin kalitesini k-NN ve linear probe ile ölçüp, önceki çalışmada
(ViT-B/16 vs Swin-T) elde edilen fine-tuned sonuçlarla karşılaştırmak.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

```bash
python dino_feature_eval.py --model facebook/dinov2-small
python dino_feature_eval.py --model facebook/dinov2-base
python dino_feature_eval.py --model facebook/dinov3-vits16-pretrain-lvd1689m
```

Script:
1. Belirtilen backbone'u ve ona ait `AutoImageProcessor`'ı yükler (model dondurulur, gradyan hesaplanmaz).
2. CIFAR-10 train/test setlerini indirir, modele uygun ön işlemeyi (resize + ImageNet normalizasyonu) uygular.
3. CLS token özelliklerini çıkarır.
4. Özellikler üzerinde k-NN (varsayılan k=20, cosine metrik) ve logistic regression tabanlı linear probe eğitir.
5. Test doğruluğunu raporlar.

Elde edilen k-NN / linear probe doğrulukları, önceki makaledeki fine-tuned
ViT-B/16 vs Swin-T sonuçlarıyla karşılaştırılmak üzere kullanılır.
