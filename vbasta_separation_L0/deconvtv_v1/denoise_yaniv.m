clear all
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo file for deconvtv
% Image 'salt and pepper' noise removal
% 
% Stanley Chan
% University of California, San Diego
% 20 Jan, 2011
%
% Copyright 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prepare images
% f_orig  = im2double(imread('./data/building.jpg'));
load('cat4.mat');
f_orig = imgc;
noiseSd = 0.01;
[rows cols frames] = size(f_orig);
H       = fspecial('gaussian', [1 1], 1);
% g       = imfilter(f_orig, H, 'circular');
g       = imnoise(f_orig, 'gaussian', 0, noiseSd);

% Setup parameters (for example)
% opts.rho_r   = 5;
% opts.rho_o   = 100;
% opts.beta    = [1 1 0];
% opts.print   = true;
% opts.alpha   = 0.7;
% opts.method  = 'l1';

opts.rho_r   = 1/(2*sqrt(noiseSd));
opts.beta    = [1 1 0];
opts.print   = true;
opts.alpha   = 0.1;
opts.method  = 'l2';


% Setup mu
mu           = 20;

% Main routine

out = deconvtv(g, H, mu, opts);

% Display results
figure(1);
imshow(g);
title('input');

figure(2);
imshow(out.f);
title('output');