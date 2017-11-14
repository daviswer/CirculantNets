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


vbity = 200
epoch = 2000
start = time.time()

runningloss = 0
for it in range(10*epoch):
    if it%10==0:
        print(it)
    
    # Build batch
    support, query, targs, _ = batchmaker(way,trainshot,testshot,trainset,alterful=True)
    
    # Predict
    embed.zero_grad()
    model.zero_grad()
    embeds = embed(support)
    qembeds = embed(query)
    preds = model(embeds,qembeds,way)

    # Calculate Loss
    loss = criterion(preds, targs)
    runningloss += loss.data[0]
    
    # Backprop
    if it%epoch == 0:
        optimizer = optim.Adam(embed.parameters(),lr=.001/(2**(it//epoch)))
    loss.backward()
#     nn.utils.clip_grad_norm(model.parameters(), 1)
    optimizer.step()
    
    # Report
    if it%vbity == vbity-1:
        display.clear_output(wait=True)

        losstracker.append(runningloss/vbity)
        embed = embed.eval()
        evalloss, evalacc, _ = evaluate(embed,model,criterion,evalway,trainshot,testshot,reps,testset)
        embed = embed.train()
        evallosstracker.append(evalloss)
        evalacctracker.append(evalacc)
        
        pl.figure(1,figsize=(15,5))
        pl.subplot(1,2,1)
        pl.plot(losstracker)
        pl.plot(evallosstracker)
        pl.ylim((.5,3))
        pl.title("Loss: Training Blue, Validation Gold")
        pl.subplot(1,2,2)
        pl.plot(evalacctracker[::-1])
        pl.ylim((0.3,.8))
        pl.title("Validation Acc")
        pl.show()
        
        print("Train loss is: "+str(runningloss/vbity)+
              "\nValidation accuracy is: "+str(evalacc)+
              "\nValidation loss is: "+str(evalloss)+"\n")
        runningloss = 0
        print(time.time()-start)
