import sys, json
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

CLASSES = ['BIRD_DROP', 'CLEAN', 'DUSTY', 'PHYSICAL_DAMAGE']
tf = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def load(weights_path):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 4)
    sd = torch.load(weights_path, map_location='cpu', weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m

def predict(model, img_path):
    im = Image.open(img_path).convert('RGB')
    x = tf(im).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0).tolist()
    idx = max(range(4), key=lambda i: probs[i])
    return CLASSES[idx], probs[idx]

baseline = load('weights/mobilenet_condition_4class_research_v1.pth')
mix1 = load('weights/mobilenet_condition_4class_rooftop_mix_oversample1_v1.pth')

# held-out rooftop TEST crops (scene-disjoint, never trained on for baseline; for mix1
# these were in the eval "test" split loader, NOT the mix training loader)
test_dir = Path('datasets/ground_pv_research/rooftop_condition_v1/crops/test')
sheet = {row.split(',')[0]: row.split(',')[4] for row in
         Path('datasets/ground_pv_research/rooftop_condition_v1/labeling_sheet.csv').read_text().splitlines()[1:]
         if row.strip()}
import csv
rows = list(csv.DictReader(open('datasets/ground_pv_research/rooftop_condition_v1/labeling_sheet.csv')))
test_rows = [r for r in rows if r['split']=='test' and r['label'] not in ('', 'SKIP')]

results = {'baseline': {'correct':0,'total':0,'pred_counts':{}}, 'mix1': {'correct':0,'total':0,'pred_counts':{}}}
mismatches = []
for r in test_rows:
    p = test_dir / f"{r['crop_id']}.jpg"
    gt = r['label']
    for tag, model in (('baseline', baseline), ('mix1', mix1)):
        pred, conf = predict(model, p)
        d = results[tag]
        d['total'] += 1
        d['correct'] += int(pred == gt)
        d['pred_counts'][pred] = d['pred_counts'].get(pred, 0) + 1
        if tag == 'baseline' and pred != gt:
            mismatches.append((r['crop_id'], gt, pred, round(conf,2)))

for tag, d in results.items():
    print(tag, 'accuracy:', d['correct'], '/', d['total'], '=', round(d['correct']/d['total'],3), ' pred_counts:', d['pred_counts'])
print()
print('baseline mismatches (gt=CLEAN mostly, all test rows are CLEAN except 2 DUSTY):')
for m in mismatches:
    print(' ', m)

mix10 = load('weights/mobilenet_condition_4class_rooftop_mix_oversample10_v1.pth')
c10 = 0
pred_counts10 = {}
for r in test_rows:
    p = test_dir / f"{r['crop_id']}.jpg"
    pred, conf = predict(mix10, p)
    c10 += int(pred == r['label'])
    pred_counts10[pred] = pred_counts10.get(pred, 0) + 1
print('mix10 accuracy:', c10, '/', len(test_rows), '=', round(c10/len(test_rows),3), 'pred_counts:', pred_counts10)
