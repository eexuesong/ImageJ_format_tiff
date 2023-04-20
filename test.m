%% Write a 8-bit 2D tiff file using default settings (resolution = 0.05 um, spacing = 0.1 um)
stack_in = uint8(255 * rand(1, 512 * 512));
stack_in = reshape(stack_in, [512, 512]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '8_bit(default_settings).tif');
toc
%% Read a 8-bit 2D tiff file
tic
stack_out = ImageJ_formatted_TIFF.ReadTifStack('8_bit(default_settings).tif');
toc
isequal(stack_in, stack_out)



%% Write a 16-bit 2D tiff file using default settings (resolution = 0.05 um, spacing = 0.1 um)
stack_in = uint16(65535 * rand(1, 512 * 512));
stack_in = reshape(stack_in, [512, 512]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '16_bit(default_settings).tif');
toc

%% Write a 32-bit 2D tiff file using default settings (resolution = 0.05 um, spacing = 0.1 um)
stack_in = single(rand(1, 512 * 512));
stack_in = reshape(stack_in, [512, 512]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '32_bit(default_settings).tif');
toc




%% Write a 8-bit 3D tiff file using user settings (resolution = 0.046 um, spacing = 0.2 um)
stack_in = uint8(linspace(0, 255, 512 * 512));
stack_in = repmat(stack_in, 1, 1, 100);
stack_in = reshape(stack_in, [512, 512, 100]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '8_bit_stack(user_setttings).tif', 0.046, 0.2);
toc




%% Write a 16-bit 3D tiff file using user settings (resolution = 0.046 um, spacing = 0.2 um)
stack_in = uint16(linspace(0, 65535, 1920 * 1280));
stack_in = reshape(stack_in, [1920, 1280]);
stack_in = repmat(stack_in, 1, 1, 100);
stack_in = reshape(stack_in, [1920, 1280, 100]);
stack_in = permute(stack_in, [2 1 3]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '16_bit_stack(user_setttings).tif', 0.046, 0.2);
toc
%% Read a 16-bit 3D tiff file and display its resolution, spacing and unit
tic
[stack_out, header] = ImageJ_formatted_TIFF.ReadTifStack('16_bit_stack(user_setttings).tif');
toc
isequal(stack_in, stack_out)
disp("Resolution: " + header.resolution)
disp("Spacing: " + header.spacing)
disp("Unit: " + header.unit)
%% Read a top 15% part and bottom 15% part 3D tiff file with downsampling factor = 8 in x and z
crop = 0.15;
ycrop = round(header.ImageLength * crop);
xcrop = round(header.ImageWidth * crop);
DownX = 8;
DownY = 1;
DownZ = 8;
top_part = ImageJ_formatted_TIFF.ReadTifStackPortion('16_bit_stack(user_setttings).tif', 'y', [1, ycrop], DownX, DownZ);
bottom_part = ImageJ_formatted_TIFF.ReadTifStackPortion('16_bit_stack(user_setttings).tif', 'y', [header.ImageLength - ycrop + 1, header.ImageLength], DownX, DownZ);
ImageJ_formatted_TIFF.WriteTifStack(top_part, 'top_downsampling.tif', 0.046, 0.2);
ImageJ_formatted_TIFF.WriteTifStack(bottom_part, 'bottom_downsampling.tif', 0.046, 0.2);

left_part = ImageJ_formatted_TIFF.ReadTifStackPortion('16_bit_stack(user_setttings).tif', 'x', [1, xcrop], DownY, DownZ);
right_part = ImageJ_formatted_TIFF.ReadTifStackPortion('16_bit_stack(user_setttings).tif', 'x', [header.ImageWidth - xcrop + 1, header.ImageWidth], DownY, DownZ);
ImageJ_formatted_TIFF.WriteTifStack(left_part, 'left_downsampling.tif', 0.046, 0.2);
ImageJ_formatted_TIFF.WriteTifStack(right_part, 'right_downsampling.tif', 0.046, 0.2);




%% Write a 16-bit 4D tiff file using user settings (resolution = 0.046 um, spacing = 0.2 um, imformat = 'cz')
stack_in = uint16(linspace(0, 65535, 1920 * 1280));
stack_in = reshape(stack_in, [1920, 1280]);
stack_in = repmat(stack_in, 1, 1, 4, 25);
stack_in = permute(stack_in, [2 1 3 4]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '16_bit_4D_stack(cz).tif', 0.046, 0.2, 'cz');
toc
%% Read a 16-bit 4D tiff file and display its channels, slices and frames
tic
[stack_out, header] = ImageJ_formatted_TIFF.ReadTifStack('16_bit_4D_stack(cz).tif');
toc
isequal(stack_in, stack_out)
disp("Channels: " + header.channels)
disp("Slices: " + header.slices)
disp("frames: " + header.frames)




%% Write a 16-bit 5D tiff file using user settings and break the 4GB limit (resolution = 0.046 um, spacing = 0.2 um, imformat = 'czt')
stack_in = uint16(linspace(0, 65535, 1920 * 1280));
stack_in = reshape(stack_in, [1920, 1280]);
stack_in = repmat(stack_in, 1, 1, 4, 100, 5);
stack_in = permute(stack_in, [2 1 3 4 5]);
tic
ImageJ_formatted_TIFF.WriteTifStack(stack_in, '16_bit_5D_stack(czt).tif', 0.046, 0.2, 'czt');
toc
%% Get the header information about a 16-bit 5D tiff file without reading any image data
tic
header = ImageJ_formatted_TIFF.get_header('16_bit_5D_stack(czt).tif');
toc



