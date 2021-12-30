import json
import numpy as np
from data import settings, LoadDataSet,DataLoaderInput
from deterministic import Deterministic_net
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from BMA import monte_carlo_bma
from KFAC import KFAC
from SWAG import Swag
from sklearn.datasets import make_moons

def uncertainty_hist(data, title):
    density, edges = np.histogram(data, bins = 50, density = True)
    fig, ax = plt.subplots()
    ax.hist(edges[:-1], edges, weights=density)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Predictive '+title+' density')
    plt.show()


from utils import plot_decision_boundary
sns.set_style('darkgrid')
# TODO: Husk at ændre i nn hvis du vil fjerne dropout!!!!
# Opening JSON file
which_model = 'swag'
which_data = 'two_moons'
test_with_temp = False
predict_func_boundary = 'predict'



if which_model.lower() == 'swag':
    # loads information from grid search
    info_path = 'SWAG/' + which_data + '/gridsearch/params_evals/info.json'

    if test_with_temp:
        temp_path = 'SWAG/' + which_data + '/calibrating/params_evals/info.json'
        f = open(temp_path)
        data = json.load(f)
        temp_test_loss = data['key'][0]['test_loss']
        opt_temp_idx = np.argmin(temp_test_loss)
        temp = data['key'][0]['temp'][opt_temp_idx]

    else:
        temp = 1

elif which_model.lower() == 'kfac':
        # loads information from grid search
    info_path = 'KFAC/' + which_data + '/gridsearch/params_evals/info.json'

    if test_with_temp:
        temp_path = 'KFAC/' + which_data + '/calibrating/params_evals/info.json'
        f = open(temp_path)
        data = json.load(f)
        temp_test_loss = data['key'][0]['test_loss']
        opt_temp_idx = np.argmin(temp_test_loss)
        temp = data['key'][0]['temp'][opt_temp_idx]
    else:
        temp = 1

elif which_model.lower() == 'deterministic'
    info_path = 'Deterministic/'+which_data+'/bo/params_evals/model_info_BO.json'
    if test_with_temp:
        temp_path = 'Deterministic/' + which_data + '/calibrating/params_evals/info.json'
        f = open(temp_path)
        data = json.load(f)
        temp_test_loss = data['key'][0]['test_loss']
        opt_temp_idx = np.argmin(temp_test_loss)
        temp = data['key'][0]['temp'][opt_temp_idx]
    else:
        temp = 1

elif which_model.lower() == 'ensample':
    info_path = 'Deterministic/two_moons/bo/params_evals/model_info_BO.json'
    if test_with_temp:
        temp_path = 'Deterministic/' + which_data + '/ensample/calibrating/params_evals/info.json'
        f = open(temp_path)
        data = json.load(f)
        temp_test_loss = data['key'][0]['test_loss']
        opt_temp_idx = np.argmin(temp_test_loss)
        temp = data['key'][0]['temp'][opt_temp_idx]

else:
    temp = 1

# read json
f = open(info_path)
data = json.load(f)
N = len(data['key'])

# find all test loss of last epoch
try:
    all_last_testloss = [data['key'][i][str(i)]['evals']['test_loss'][-1] for i in range(N)]
except:
    all_last_testloss = [data['key'][i][str(i)]['evals:']['test_loss'][-1] for i in range(N)]

# find optimal model settings
opti_test_loss = min(all_last_testloss)
argmin = np.argmin(all_last_testloss)
opti_params = data['key'][argmin][str(argmin)]['params']

# in and output dimensions
input_dim, output_dim = settings(which_data)

# Define paths to load models from
if which_data == 'two_moons':
    model_path = 'Deterministic/' + which_data + '/bo/models/m_' + str(int(argmin))
else:
    # use model trained with dropout
    model_path = 'Deterministic/' + which_data + '/optimal/model' + str(int(argmin))

path_to_bma_SWAG = "SWAG/" + which_data + '/' + 'gridsearch/bma/idx_models_' + str(argmin) + 'temp_1/'
path_to_bma_KFAC = "KFAC/" + which_data + '/' + 'gridsearch/bma/idx_models_' + str(argmin) + 'temp_1/'
path_to_bma_ensample = "Deterministic/" + which_data + '/ensample/model/'


# Load data
if which_data == 'two_moons':
    D = test_data = make_moons(n_samples=1000, noise=0.3, random_state=10)
    Xtest, ytest = torch.tensor(D[0], dtype=torch.float32), torch.tensor(D[1])
    dataset = DataLoaderInput(Xtest, ytest, which_data=which_data)
else:
    D = LoadDataSet(dataset=which_data)
    Xtrain, ytrain, Xtest, ytest = D.load_data_for_CV()
    dataset = DataLoaderInput(Xtest, ytest, which_data=which_data)
test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=1,
                                          shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

# Load model and test on test set
if which_model == 'deterministic':
    Net = Deterministic_net(input_dim, int(opti_params['hidden_dim']), output_dim)
    Net.load_state_dict(torch.load(model_path))

    # compute accuracy, average loss, all probs, average loss with reg, all loss
    accuracy, loss_ave, all_probs, loss_l2_ave, all_loss = Net.test_net(test_loader=test_loader, criterion=criterion,
                                                                        temp=temp, l2=opti_params['l2'],  test = True)

elif which_model.lower() == 'kfac':
    Net = KFAC(input_dim = input_dim, hidden_dim = int(opti_params['hidden_dim']), output_dim = output_dim, momentum = opti_params['momentum'], l2_param = opti_params['l2'], L = 3)
    p_yxw, all_probs, all_loss, accuracy, loss_l2 = monte_carlo_bma(Net, Xtest, ytest, S = 30, C =output_dim, temp = temp, criterion = criterion, path_to_bma=path_to_bma_KFAC,
                   save_probs='', batch =  True)


elif which_model.lower() == 'swag':
    Net = Swag(input_dim = input_dim, hidden_dim = int(opti_params['hidden_dim']), output_dim = output_dim, K = 20, c = 1, S = 30, criterion = criterion, l2_param = opti_params['l2'])

    p_yxw, all_probs, all_loss, accuracy, loss_l2  = monte_carlo_bma(Net, Xtest, ytest, S = 30, C =output_dim, temp = temp, criterion = criterion, path_to_bma=path_to_bma_SWAG,
                       save_probs='', l2 = opti_params['l2'], batch =  True)


elif which_model.lower() == 'ensample':
    Net = Deterministic_net(input_dim = input_dim, hidden_dim = int(opti_params['hidden_dim']) , output_dim = output_dim)
    p_yxw, all_probs, all_loss, accuracy, loss_l2 = monte_carlo_bma(model = Net, Xtest = Xtest, ytest = ytest, S = 5, C  = output_dim, temp = temp,
                                                                    criterion = criterion, l2 = opti_params['l2'], forplot=False, path_to_bma= path_to_bma_ensample, batch=True, which_data=which_data)



# Compute standard error of nll loss (CLT says you can do it)
C= len(all_probs[0])
N = len(all_loss)
s = torch.std(torch.tensor(all_loss))
SE = s/torch.sqrt(torch.tensor(N))
nll = sum(all_loss)/N

# Reliability diagrams and ECE
M = 20
I = [((m-1)/M, m/M) for m in range(1,M+1)]
I[-1] = (I[-1][0],I[-1][0]+0.1)
Counter = {i:{'correct': 0, 'incorrect':0, 'probs': []} for i in I}

for probs,y in zip(all_probs,dataset.y):
    correct = probs[y]
    rest = torch.cat((probs[0:y], probs[y:-1]), 0)
    for key in Counter.keys():
        if correct >= key[0] and correct < key[1]:
            Counter[key]['correct'] +=1
            Counter[key]['probs'].append(correct.item())
        for p in rest:
            if p >= key[0] and p < key[1]:
                Counter[key]['incorrect'] += 1
                Counter[key]['probs'].append(p.item())

acc = [Counter[i]['correct']/(Counter[i]['correct'] +Counter[i]['incorrect']) for i in Counter.keys()]
conf = [sum(Counter[i]['probs'])/(Counter[i]['correct'] +Counter[i]['incorrect']) for i in Counter.keys()]

# Expected calibration error
ece_all= [(Counter[key]['correct'] +Counter[key]['incorrect'])/(N*C) * abs(acc[i] - conf[i]) for i,key in enumerate(Counter.keys())]
ece = sum(ece_all)
ece_se = torch.std(torch.tensor(ece_all))/torch.sqrt((torch.tensor(N*C)))

#  Reliability diagrams
fig, ax = plt.subplots()

bins = [key[0] for key in Counter.keys()] +[1]
ax.hist(bins[:-1], bins, weights=acc)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.set_xlabel('Confidence')
ax.set_ylabel('Accuracy')
props = dict(facecolor='gray', alpha=0.6)
txt_string = 'ECE: {}'.format(ece)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, txt_string, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.show()

# Entropy/total uncertainty of predictive distribution (H)
H = -torch.sum(torch.log2(all_probs)*all_probs,1)
uncertainty_hist(H, 'entropy uncertainty')
if which_model == 'determnistic':
    pass
else:
    aleatoric = torch.zeros(len(all_loss))
    for val in p_yxw.values():
        aleatoric += torch.sum(torch.log2(val) * val, 1)
    aleatoric = -aleatoric/len(p_yxw)

    epistemic = H - aleatoric

    uncertainty_hist(aleatoric, 'aleatoric uncertainty')
    uncertainty_hist(aleatoric, 'epistemic uncertainty')
    pass



# decision boundary
#plot_decision_boundary(Net, test_loader, S = 30, temp = temp, path_to_bma=path_to_bma, title="", predict_func = predict_func_boundary, save_image_path = "Deterministic/two_moons/plt.jpg", sample_new_weights = False)

