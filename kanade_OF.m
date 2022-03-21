




vidReader = VideoReader('/home/morgoth/computer_vision_prototype/CV_output/with_resize/gauss_smol.mp4');
save_path = '/home/morgoth/computer_vision_prototype/CV_output/small_gauss_OF/'


h = gcf; % was figure
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);
opticFlow = opticalFlowLK ;
i = 1 

while hasFrame(vidReader)
    i = i + 1
    frame = readFrame(vidReader);
    frame = im2gray(frame) ;
    imshow(frame)
    jpgFileName = strcat(save_path, num2str(i), '.jpg');
    
    flow = estimateFlow(opticFlow, frame);

    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',3,'Parent',hPlot);
    hold off 
   
    saveas(h,jpgFileName)
    %exportgraphics(h,jpgFileName ,'Resolution',300)
end