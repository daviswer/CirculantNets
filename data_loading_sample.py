standardize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
            ])
alter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(84,padding=10),
            standardize
            ])

def batchmaker(way,trainshot,testshot,theset,alterful=False):
    classes = np.random.choice(len(theset),way)
    if alterful:
        li = [torch.cat([alter(theset[cl][i]).view(1,3,84,84) for i in 
                         np.random.choice(600,trainshot+testshot)],dim=0).float()
              for cl in classes]
    else:
        li = [torch.cat([standardize(theset[cl][i]).view(1,3,84,84) for i in 
                         np.random.choice(600,trainshot+testshot)],dim=0).float()
              for cl in classes]
    support = torch.cat([t[:trainshot,:,:,:] for t in li],dim=0)
    stargs = torch.LongTensor([i//trainshot for i in range(trainshot*way)])
    query = torch.cat([t[trainshot:,:,:,:] for t in li],dim=0)
    qtargs = torch.LongTensor([i//testshot for i in range(testshot*way)])
    return(Variable(support).cuda(),
           Variable(query, volatile=(not alterful)).cuda(),
           Variable(qtargs, volatile=(not alterful)).cuda(),
           Variable(stargs).cuda()
          )
