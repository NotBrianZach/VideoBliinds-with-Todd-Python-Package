filenames = dir('/scratch/01891/adb/ref_video_test/*.bmp'); %# get information of all .bmp files in work dir
n  = numel(filenames);    %# number of .bmp files

% for i = 1:n   C:\Users\Zach\Desktop\movieFrames\*
%     A = imread( filenames(i).name );
% 
%     %# gets full path, filename radical and extension
%     [fpath radical ext] = fileparts( filenames(i).name ); 
% 
%     save([radical '.mat'], 'A');                          
% end

for i = 1:n %/200
    %C:\Users\Zach\Desktop\movieFrames\
    im = imread( ['/Users/brian/Desktop/VideoBliinds/' filenames(i).name] );
    frames(:,:,i) = im(:,:,1)
    %# gets full path, filename radical and extension
    [fpath radical ext] = fileparts( filenames(i).name ); 
end
    save(['frames.mat'], 'im');                          

% x = load([radical);
% y = load("filename1.mat"_);
% save('combine.mat', 'x', 'y');






%filenames = dir('/Users/brian/Desktop/VideoBliinds/*.bmp'); %# get information of all .bmp files in work dir
%n  = numel(filenames);    %# number of .bmp files
%
%% for i = 1:n   C:\Users\Zach\Desktop\movieFrames\*
%%     A = imread( filenames(i).name );
%% 
%%     %# gets full path, filename radical and extension
%%     [fpath radical ext] = fileparts( filenames(i).name ); 
%% 
%%     save([radical '.mat'], 'A');                          
%% end
%
%for i = 1:49%n %/200
%    %C:\Users\Zach\Desktop\movieFrames\
%    im = imread( ['/Users/brian/Desktop/VideoBliinds/' filenames(i).name] );
%    frames(:,:,i) = im(:,:,1)
%    %# gets full path, filename radical and extension
%    [fpath radical ext] = fileparts( filenames(i).name ); 
%end
%    save(['frames.mat'], 'im');                          
%
%% x = load([radical);
%% y = load("filename1.mat"_);
%% save('combine.mat', 'x', 'y');
