import torch
import faulthandler

from code.model import DeepMarchingCube
from code.loader import get_loader
from code.visualize import visualize

if __name__ == '__main__':
    faulthandler.enable()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8

    test_loader = get_loader(set='test', batch_size=batch_size)

    model = DeepMarchingCube()
    model.load_state_dict(torch.load('outputs/models/best_model.pth'))
    model.eval()

    print('Now visualizing...')
    avg_chamfer_dist, avg_hamming_dist = visualize(model, test_loader, device)
    print('Done!')

    print(f'Average Chamfer Distance: {avg_chamfer_dist}')
    print(f'Average Hamming Distance: {avg_hamming_dist}')