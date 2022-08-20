import numpy as np
from scipy.io import savemat
import os

weights = []
net_dims = [784, 100, 50, 10]
num_layers = len(net_dims) - 1
norm_const = 1 / np.sqrt(num_layers)

for i in range(1, len(net_dims)):
  weights.append(norm_const * np.random.rand(net_dims[i], net_dims[i-1]))


# from a FNN model
weights = []
net.eval()
for i in range(5):
    keyword = "fc" + np.array2string(np.array(i + 1)) + ".weight"
    weights.append(np.float64(net.state_dict()[keyword].numpy()))

fname = os.path.join(os.getcwd(), 'data/weights/random_weights.mat')
data = {'weights': np.array(weights, dtype=np.object)}
savemat(fname, data)



# # on command line
# # on LipSDP/LipSDP directory
# python LipSDP/LipSDP/solve_sdp.py --form layer --weight-path data/weights/random_weights.mat




# start_time = time()
eng = matlab.engine.start_matlab()
eng.addpath(r'matlab_engine')
eng.addpath(r'matlab_engine/weight_utils')
eng.addpath(r'matlab_engine/error_messages')
eng.addpath(r'examples/saved_weights')

network = {
    'alpha': matlab.double([args.alpha]),
    'beta': matlab.double([args.beta]),
    'weight_path': args.weight_path,
}

lip_params = {
    'formulation': args.form,
    'split': matlab.logical([args.split]),
    'parallel': matlab.logical([args.parallel]),
    'verbose': matlab.logical([args.verbose]),
    'split_size': matlab.double([args.split_size]),
    'num_neurons': matlab.double([args.num_neurons]),
    'num_workers': matlab.double([args.num_workers]),
    'num_dec_vars': matlab.double([args.num_decision_vars])
}


start_time = time()
L = eng.solve_LipSDP(network, lip_params, nargout=1)
print(f'LipSDP-{args.form.capitalize()} gives a Lipschitz constant of {L:.3f}')
print(f'Total time: {float(time() - start_time):.5} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--form',
        default='neuron',
        const='neuron',
        nargs='?',
        choices=('neuron', 'network', 'layer', 'network-rand', 'network-dec-vars'),
        help='LipSDP formulation to use')

    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='prints CVX output from solve if supplied')

    parser.add_argument('--alpha',
        type=float,
        default=0,
        nargs=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--beta',
        type=float,
        default=1,
        nargs=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--num-neurons',
        type=int,
        default=100,
        nargs=1,
        help='number of neurons to couple for LipSDP-Network-rand formulation')

    parser.add_argument('--split',
        action='store_true',
        help='splits network into subnetworks for more efficient solving if supplied')

    parser.add_argument('--parallel',
        action='store_true',
        help='parallelizes solving for split formulations if supplied')

    parser.add_argument('--split-size',
        type=int,
        default=2,
        nargs=1,
        help='number of layers in each subnetwork for splitting formulations')

    parser.add_argument('--num-workers',
        type=int,
        default=0,
        nargs=1,
        help='number of workers for parallelization of splitting formulations')

    parser.add_argument('--num-decision-vars',
        type=int,
        default=10,
        nargs=1,
        help='specify number of decision variables to be used for LipSDP')

    parser.add_argument('--weight-path',
        type=str,
        required=True,
        nargs=1,
        help='path of weights corresponding to trained neural network model')

    args = parser.parse_args()

    if args.parallel is True and args.num_workers[0] < 1:
        raise ValueError('When you use --parallel, --num-workers must be an integer >= 1.')

    if args.split is True and args.split_size[0] < 1:
        raise ValueError('When you use --split, --split-size must be an integer >= 1.')






