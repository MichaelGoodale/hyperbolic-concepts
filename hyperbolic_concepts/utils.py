import torch 
import os

PREDICATE_PATH = (os.path.join(os.path.dirname(__file__), "../predicates.txt"))
CLASS_PATH = (os.path.join(os.path.dirname(__file__), "../classes.txt"))

feature2idx = {}
features = []
with open(PREDICATE_PATH) as f:
  for l in f:
    idx, feature = l.strip().split('\t')
    idx = int(idx) - 1
    feature2idx[feature] = idx
    features.append(feature)
classes2idx = {}
dumb_class_map = {}
classes = []
with open(CLASS_PATH) as f:
  for l in f:
    idx, class_name = l.strip().split('\t')
    dumb_class_map[class_name] = int(idx) - 1
    classes.append(class_name)
classes.sort()
for i, k in enumerate(classes):
  classes2idx[k] = i
  dumb_class_map[i] = dumb_class_map[k]
features = ['furry', 'hairless', 'big', 'small', 'hands', 'hooves', 'paws', 'longleg', 'tail', 'longneck', 'smelly',
            'flys', 'swims', 'walks', 'fast', 'slow', 'bipedal', 'quadrapedal', 'fish', 'meat', 'plankton', 'vegetation', 'nocturnal']
features_idx = [feature2idx[f] for f in features]



super_animal={'whale':['blue+whale', 'humpback+whale'],
              'dolphins': ['killer+whale', 'dolphin'],
              'cetaceans': ['whale', 'dolphins'],
              'pinniped': ['seal', 'walrus'],
              'sea+mammals': ['cetaceans', 'pinniped'],
              'bear': ['grizzly+bear', 'giant+panda', 'polar+bear'],
              'horses': ['horse', 'zebra'],
              'rodent': ['mouse', 'rat', 'hamster', 'squirrel', 'beaver'],
              'big+cat': ['lion', 'tiger', 'leopard'],
              'house+cat':['persian+cat', 'siamese+cat'],
              'cat':['house+cat', 'big+cat', 'bobcat'],
              'cattle': ['ox', 'cow'],
              'bovine': ['buffalo', 'cattle'],
              'bovid': ['antelope', 'bovine'],
              'cervidae': ['moose', 'deer'],
              'mustelidae':['otter', 'weasel'],
              'canine': ['dog', 'wolf'],
              'canid': ['canine', 'fox'],
              'dog': ['collie', 'chihuahua', 'german+shepherd', 'dalmatian'],
              'ape': ['gorilla', 'chimpanzee'],
              'primate': ['ape', 'spider+monkey']
              } 

super_animals_2_idx = {}
classes2membership = {k: set() for k in classes2idx}

def recursive_name_def(name, members):
  if name in classes2idx:
    classes2membership[name].update(members)
  else:
    if name not in super_animals_2_idx:
      super_animals_2_idx[name] = len(super_animals_2_idx)
    members.append(name)
    for sub_name in super_animal[name]:
      recursive_name_def(sub_name, members.copy())

for k in super_animal:
  recursive_name_def(k, [])

super_animal_mat = torch.zeros(len(classes2idx), len(super_animals_2_idx))
for k in classes2membership:
  supersets = classes2membership[k]
  for superset in supersets:
    super_idx = super_animals_2_idx[superset]
    super_animal_mat[classes2idx[k], super_idx] = 1
