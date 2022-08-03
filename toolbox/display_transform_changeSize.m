function display_transform_changeSize(mOmega, tranSize)
    for i=1:size(mOmega,1)
        mOmega(i,:)=mOmega(i,:)-min(mOmega(i,:));
        if(max(mOmega(i,:))>0)
          mOmega(i,:)=mOmega(i,:)/(max(mOmega(i,:)));
        end
    end

    jy=1;cc=1;
    Ta=(max(max(mOmega)))*ones((tranSize+jy)*(tranSize-1) + tranSize,(tranSize+jy)*(tranSize-1) + tranSize);
    for i=1:tranSize+jy:((tranSize-1)*(tranSize+jy))+1
        for j=1:tranSize+jy:((tranSize-1)*(tranSize+jy))+1
           Ta(i:i+(tranSize-1),j:j+(tranSize-1))=reshape((mOmega(cc,:))',tranSize,tranSize);
           cc=cc+1;
        end
    end
    imagesc(Ta);colormap('Gray');axis off;axis image; drawnow
end