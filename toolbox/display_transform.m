function display_transform(mOmega)
    for i=1:size(mOmega,1)
        mOmega(i,:)=mOmega(i,:)-min(mOmega(i,:));
        if(max(mOmega(i,:))>0)
          mOmega(i,:)=mOmega(i,:)/(max(mOmega(i,:)));
        end
    end

    jy=1;cc=1;
    Ta=(max(max(mOmega)))*ones((8+jy)*7 + 8,(8+jy)*7 + 8);
    for i=1:8+jy:(7*(8+jy))+1
        for j=1:8+jy:(7*(8+jy))+1
           Ta(i:i+7,j:j+7)=reshape((mOmega(cc,:))',8,8);
           cc=cc+1;
        end
    end
    imagesc(Ta);colormap('Gray');axis off;axis image; drawnow
end