classdef ImageJ_formatted_TIFF
    % Reads and writes tiff stack following ImageJ format pseudo-Tiff.

    % V2.0 Xuesong Li 04/13/2023:
    % (1) Removes function "ReadTifStack_v2" and integrates its feature into
    % "ReadTifStack" to increase the read speed.
    % (2) Adds read compatibility of conventional tiff stack and
    % Micro-manager tiff stack.
    % (3) When calling "ReadTifStack", user now can choose to output
    % "Header" to know more properties of the image stack.
    % (4) Updates "Simple_IFD.m" and adds the following properties:
    %   (i) IJMetadataByteCounts, IJMetadata
    %   (ii) NumberCharsInOMEXMLMetadata, OffsetOfOMEXMLMetadata, OMEXMLMetadata
    %   (iii) NumberCharsInMicroManagerMetadata,
    %   OffsetOfOMicroManagerMetadata,MicroManagerMetadata
    %   (iv) resolution, images, channels (c), slices (z), frames (t),
    %   unit, spacing
    
    
    % V1.5 Yicong Wu 09/01/2022:    Add functions: "ReadTifStack_v2",
    % "ReadTifStackOneSlice" and "ReadTifStackPortion"

    % V1.0 Xuesong Li 01/01/2020:
    

    methods(Static)
        function WriteTifStack(Stack, Filename, resolution, spacing, imformat)
            % imformat is image format (string): can be 'c','z','t', 'cz',
            % 'ct', 'zt' or 'czt'

            if nargin == 2
                resolution = 0.05;  % Default lateral pixel size: 50 nm
                spacing = 0.1;      % Default axial pixel size: 100 nm
                imformat = 'z';
            elseif nargin == 3
                spacing = 0.1;
                imformat = 'z';
            elseif nargin == 4
                imformat = 'z';
            end

            fileID = fopen(Filename, 'w+');
            dimension = size(Stack);
            Height = dimension(1);
            Width = dimension(2);
            dimension_czt = dimension(3:end);
            
            channels = 1;
            slices = 1;
            frames = 1;
            if size(dimension_czt, 2) == 0
                % 2D slice
                Stack = permute(Stack, [2 1]);
            else
                % 3D, 4D or 5D stack
                if size(dimension_czt, 2) ~= size(imformat, 2)
                    error("'imformat' dose not match the input Stack dimension.")
                end
                switch size(imformat, 2)
                    case 1
                        if imformat == 'c'
                            channels = dimension_czt(1);
                        elseif imformat == 'z'
                            slices = dimension_czt(1);
                        elseif imformat == 't'
                            frames = dimension_czt(1);
                        else
                            error("'imformat' must be 'c','z','t','cz','ct','zt' or 'czt'.")
                        end
                        Stack = permute(Stack, [2 1 3]);
                    case 2
                        if strcmp(imformat, 'cz')
                            channels = dimension_czt(1);
                            slices = dimension_czt(2);
                        elseif strcmp(imformat, 'ct')
                            channels = dimension_czt(1);
                            frames = dimension_czt(2);
                        elseif strcmp(imformat, 'zt')
                            slices = dimension_czt(1);
                            frames = dimension_czt(2);
                        else
                            error("'imformat' must be 'c','z','t','cz','ct','zt' or 'czt'.")
                        end
                        Stack = permute(Stack, [2 1 3 4]);
                    case 3
                        if strcmp(imformat, 'czt')
                            channels = dimension_czt(1);
                            slices = dimension_czt(2);
                            frames = dimension_czt(3);
                        else
                            error("'imformat' must be 'c','z','t','cz','ct','zt' or 'czt'.")
                        end
                        Stack = permute(Stack, [2 1 3 4 5]);
                    otherwise
                        error("'imformat' has incorrect character number.")
                end
            end
            

            minval = min(Stack, [], 'all');
            maxval = max(Stack, [], 'all');
            header = ImageJ_formatted_TIFF.write_IFD(fileID, Width, Height, class(Stack), channels, slices, frames, spacing, minval, maxval, resolution);
%             Stack = permute(Stack, [2 1 3]);
            Stack = reshape(Stack, [1, Height * Width * channels * slices * frames]);

            [~, ~, system_endian] = computer;
            if system_endian ~= header.endian
                Stack = swapbytes(Stack);
            end
            Stack = typecast(Stack, 'uint8');
            fwrite(fileID, Stack, 'uint8');
            fclose(fileID);
        end

        function [Stack, Header] = ReadTifStack(Filename)
            % First determine whether it is a ImageJ formatted TIFF file (New version)
            image_info = imfinfo(Filename);
            if size(image_info, 1) > 1
%                 disp('Not a ImageJ formatted Tiff file.');
                Depth = size(image_info, 1);
                ImageJ_formatted_TIFF_flag = 0;
            else
                ImageJ_formatted_TIFF_flag = 1;
            end

            % Read and parse tiff header first
            fileID = fopen(Filename, 'r');
            header = ImageJ_formatted_TIFF.parse_tif(fileID, 0);
            ImageWidth = double(header.ImageWidth);
            ImageLength = double(header.ImageLength);
            BitsPerSample = double(header.BitsPerSample);
            Depth_estimated = floor(image_info(1).FileSize / (ImageWidth * ImageLength * BitsPerSample / 8));

            % Calculate slices from header.ImageDescription (Old version)
%             ImageDescription = convertStringsToChars(header.ImageDescription);
%             k1 = strfind(ImageDescription, 'images=');
%             ImageDescription_crop = ImageDescription(k1:end);
%             k2 = strfind(ImageDescription_crop, newline);
%             if ~isempty(k1) && ~isempty(k2)
%                 Depth = str2double(ImageDescription(k1(1) + 7: k1(1) + k2(1) - 2));
%             else
%                 warning("Did not find 'images=' in ImageDescription. Try to calculate it from filesize.");
%                 FileAttributes = dir(Filename);
%                 Depth = floor(FileAttributes.bytes / (ImageWidth * ImageLength * (header.BitsPerSample / 8)));
%             end
            
            % Calculate data_type from header.BitsPerSample
            data_type = [header.SampleFormat, header.BitsPerSample];
            if isequal(data_type, [1, 8])
                dtype = 'uint8';
            elseif isequal(data_type, [1, 16])
                dtype = 'uint16';
            elseif isequal(data_type, [1, 32])
                dtype = 'uint32';
            elseif isequal(data_type, [1, 64])
                dtype = 'uint64';
            elseif isequal(data_type, [2, 8])
                dtype = 'int8';
            elseif isequal(data_type, [2, 16])
                dtype = 'int16';
            elseif isequal(data_type, [2, 32])
                dtype = 'int32';
            elseif isequal(data_type, [2, 64])
                dtype = 'int64';
            elseif isequal(data_type, [3, 32])
                dtype = 'single';
            elseif isequal(data_type, [3, 64])
                dtype = 'double';
            else
                error("ReadTifStack does not support SampleFormat:%d with BitsPerSample: %d", header.SampleFormat, header.BitsPerSample);
            end
            
            if ImageJ_formatted_TIFF_flag
                % Old version
%                 % Initialize reading buffer and parameters
%                 PixelNum = ImageWidth * ImageLength;
%                 ByteCounts = fix(PixelNum * BitsPerSample / 8);
%                 Stack = zeros([1, Depth * PixelNum], dtype);
% 
%                 fseek(fileID, header.StripOffsets, 'bof');
%                 for i = 1:Depth
%                     Stack((i - 1) * PixelNum + 1 : i * PixelNum) = typecast(transpose(fread(fileID, ByteCounts, 'uint8=>uint8')), dtype);
%                 end

                % New version
                % Initialize reading buffer and parameters
                if isempty(header.images)
                    Depth = Depth_estimated;
                else
                    Depth = header.images;
                    % Double check
                    if Depth ~= Depth_estimated
                        warning("Image number calculated from filesize is inconsistent with 'images=' in ImageDescription.")
                    end
                end

                PixelNum = ImageWidth * ImageLength;
                ByteCounts = fix(PixelNum * BitsPerSample / 8);                
                fseek(fileID, header.StripOffsets, 'bof');
                Stack = typecast(fread(fileID, Depth * ByteCounts, 'uint8=>uint8'), dtype);

                [~, ~, system_endian] = computer;
                if system_endian ~= header.endian
                    Stack = swapbytes(Stack);
                end
            
                % Old version
%                 Stack = reshape(Stack, [PixelNum, 1, Depth]);

                Stack = reshape(Stack, [ImageWidth, ImageLength, Depth]);
                Stack = permute(Stack, [2 1 3]);
            else
                Stack = zeros([ImageLength, ImageWidth, Depth], dtype);
                for i = 1:Depth
                    Stack(:, :, i) = imread(Filename, i);
                end
            end

            fclose(fileID);

            % Further reshape Stack based on "channels", "slices" and "frames"
            if isempty(header.channels)
                channels = 1;
            else
                channels = header.channels;
            end

            if isempty(header.slices)
                slices = 1;
            else
                slices = header.slices;
            end

            if isempty(header.frames)
                frames = 1;
            else
                frames = header.frames;
            end

            if Depth ~= (channels * slices * frames)
                error("channels * slices * frames dose not match total image number.");
            end
            % Reshape into final format
            Stack = reshape(Stack, [ImageLength, ImageWidth, channels, slices, frames]);
            % Remove unnecessary dimension(s)
            Stack = squeeze(Stack);            

            if nargout > 1
                Header = header;
            end
        end
        
        function [Stack, Header] = ReadTifStackOneSlice(Filename, ZSlice)
            % First determine whether it is a ImageJ formatted TIFF file
            image_info = imfinfo(Filename);
            if size(image_info, 1) > 1
                ImageJ_formatted_TIFF_flag = 0;
            else
                ImageJ_formatted_TIFF_flag = 1;
            end

            % Read and parse tiff header first
            fileID = fopen(Filename, 'r');
            header = ImageJ_formatted_TIFF.parse_tif(fileID, 0);
            ImageWidth = double(header.ImageWidth);
            ImageLength = double(header.ImageLength);
            BitsPerSample = double(header.BitsPerSample);

            % Calculate data_type from header.BitsPerSample
            data_type = [header.SampleFormat, header.BitsPerSample];
            if isequal(data_type, [1, 8])
                dtype = 'uint8';
            elseif isequal(data_type, [1, 16])
                dtype = 'uint16';
            elseif isequal(data_type, [1, 32])
                dtype = 'uint32';
            elseif isequal(data_type, [1, 64])
                dtype = 'uint64';
            elseif isequal(data_type, [2, 8])
                dtype = 'int8';
            elseif isequal(data_type, [2, 16])
                dtype = 'int16';
            elseif isequal(data_type, [2, 32])
                dtype = 'int32';
            elseif isequal(data_type, [2, 64])
                dtype = 'int64';
            elseif isequal(data_type, [3, 32])
                dtype = 'single';
            elseif isequal(data_type, [3, 64])
                dtype = 'double';
            else
                error("ReadTifStackOneSlice does not support SampleFormat:%d with BitsPerSample: %d", header.SampleFormat, header.BitsPerSample);
            end
            
            if ImageJ_formatted_TIFF_flag
                PixelNum = ImageWidth * ImageLength;
                ByteCounts = fix(PixelNum * BitsPerSample / 8);                
                fseek(fileID, header.StripOffsets + ByteCounts * (ZSlice - 1), 'bof');
                Stack = typecast(fread(fileID, ByteCounts, 'uint8=>uint8'), dtype);

                [~, ~, system_endian] = computer;
                if system_endian ~= header.endian
                    Stack = swapbytes(Stack);
                end

                Stack = reshape(Stack, [ImageWidth, ImageLength]);
                Stack = permute(Stack, [2 1]);
            else
                Stack = imread(Filename, ZSlice);
            end

            fclose(fileID);

            if nargout > 1
                Header = header;
            end
        end

        function [Stack, Header] = ReadTifStackPortion(Filename, crop_direction, range, DownXY, DownZ)
            % "crop_direction": 'x' or 'y'
            % "range": then crop range. For example, we have a 512(x) x 
            % 512(y) image stack and we want to crop along x direction to
            % have a 64(x) x 512(y), "range" can be set as [1, 64] or [449, 512]
            % "DownXY": down-sampling factor along x dimension if crop
            % along y dimension and vice versa.
            % "DownZ": down-sampling factor along z axis


            % First determine whether it is a ImageJ formatted TIFF file
            image_info = imfinfo(Filename);
            if size(image_info, 1) > 1
                Depth = size(image_info, 1);
                ImageJ_formatted_TIFF_flag = 0;
            else
                ImageJ_formatted_TIFF_flag = 1;
            end

            % Read and parse tiff header first
            fileID = fopen(Filename, 'r');
            header = ImageJ_formatted_TIFF.parse_tif(fileID, 0);
            ImageWidth = double(header.ImageWidth);
            ImageLength = double(header.ImageLength);
            BitsPerSample = double(header.BitsPerSample);
            Depth_estimated = floor(image_info(1).FileSize / (ImageWidth * ImageLength * BitsPerSample / 8));

            % Calculate data_type from header.BitsPerSample
            data_type = [header.SampleFormat, header.BitsPerSample];
            if isequal(data_type, [1, 8])
                dtype = 'uint8';
            elseif isequal(data_type, [1, 16])
                dtype = 'uint16';
            elseif isequal(data_type, [1, 32])
                dtype = 'uint32';
            elseif isequal(data_type, [1, 64])
                dtype = 'uint64';
            elseif isequal(data_type, [2, 8])
                dtype = 'int8';
            elseif isequal(data_type, [2, 16])
                dtype = 'int16';
            elseif isequal(data_type, [2, 32])
                dtype = 'int32';
            elseif isequal(data_type, [2, 64])
                dtype = 'int64';
            elseif isequal(data_type, [3, 32])
                dtype = 'single';
            elseif isequal(data_type, [3, 64])
                dtype = 'double';
            else
                error("ReadTifStackPortion does not support SampleFormat:%d with BitsPerSample: %d", header.SampleFormat, header.BitsPerSample);
            end

            if ImageJ_formatted_TIFF_flag
                % Initialize reading buffer and parameters
                if isempty(header.images)
                    Depth = Depth_estimated;
                else
                    Depth = header.images;
                    % Double check
                    if Depth ~= Depth_estimated
                        warning("Image number calculated from filesize is inconsistent with 'images=' in ImageDescription.")
                    end
                end
                Depth = ceil(Depth / DownZ);
                
                if crop_direction == 'x'
                    x_start = range(1);
                    if range(2) == Inf
                        x_end = ImageWidth;
                    else
                        x_end = range(2);
                    end
                    PixelNum = ImageWidth * ImageLength;    % PixelNumber is the whole x-y image.
                    offset_xy = 0;                          % fread starts at the origin (left-right corner) of the x-y image.
                    offset_z = fix(ImageWidth * ImageLength * BitsPerSample / 8) * (DownZ - 1); % Always move to the origin (left-right corner) of x-y images
                elseif crop_direction == 'y'
                    y_start = range(1);
                    if range(2) == Inf
                        y_end = ImageLength;
                    else
                        y_end = range(2);
                    end
                    PixelNum = (y_end - y_start + 1) * ImageWidth;  % PixelNumber is the crop region along y dimension.
                    offset_xy = fix((y_start - 1) * ImageWidth * BitsPerSample / 8); % fread starts at one pixel on left edge.
                    offset_xy1 = fix(PixelNum * BitsPerSample / 8);
                    offset_z = fix(ImageWidth * ImageLength * BitsPerSample / 8) * DownZ - offset_xy1;
                else
                    error("Invalid crop direction.");
                end
                
                ByteCounts = fix(PixelNum * BitsPerSample / 8);
                Stack = zeros([Depth * PixelNum, 1], dtype);
                

                fseek(fileID, header.StripOffsets + offset_xy, 'bof');
                for i = 1:Depth
                    % Note: crop along x dimension is slow because we need
                    % to read the entire x-y image data within the loop
                    % each time before cropping.
                    Stack((i - 1) * PixelNum + 1 : i * PixelNum) = typecast(fread(fileID, ByteCounts, 'uint8=>uint8'), dtype);            
                    fseek(fileID, offset_z, 'cof');                
                end

                [~, ~, system_endian] = computer;
                if system_endian ~= header.endian
                    Stack = swapbytes(Stack);
                end

                if crop_direction == 'x'
                    Stack = reshape(Stack, [ImageWidth, ImageLength, Depth]);
                    Stack = Stack(x_start:x_end, 1:DownXY:end, :);
                elseif crop_direction == 'y'
                    Stack = reshape(Stack,[ImageWidth, y_end - y_start + 1, Depth]);
                    Stack = Stack(1:DownXY:end, :, :);
                end
                Stack = permute(Stack, [2 1 3]);

            else
                Depth = ceil(Depth / DownZ);
                if crop_direction == 'x'
                    x_start = range(1);
                    if range(2) == Inf
                        x_end = ImageWidth;
                    else
                        x_end = range(2);
                    end
                elseif crop_direction == 'y'
                    y_start = range(1);
                    if range(2) == Inf
                        y_end = ImageLength;
                    else
                        y_end = range(2);
                    end
                else
                    error("Invalid crop direction.");
                end

                Stack = zeros([ImageLength, ImageWidth, Depth], dtype);
                for i = 1:Depth
                    Stack(:, :, i) = imread(Filename, (i - 1) * DownZ + 1);
                end

                if crop_direction == 'x'
                    Stack = Stack(1:DownXY:end, x_start:x_end, :);
                elseif crop_direction == 'y'
                    Stack = Stack(y_start:y_end, 1:DownXY:end, :);
                end
            end
            
            fclose(fileID);

            if nargout > 1
                Header = header;
            end
        end

        function ifd = write_IFD(fileID, width, height, data_type, channels, slices, frames, spacing, min, max, resolution)
            %         We'll structure our TIF in the following way (same as how ImageJ saves tiff stack):
            %         8-byte Image File Header
            %         1st image file directory (IFD)
            %         Image description (~100 bytes)
            %         Image XResolution (Two 32-bit unsigned integers, 8 bytes)
            %         Image YResolution (Two 32-bit unsigned integers, 8 bytes)
            %         1st image data
            %         2nd image data
            %         ...
            %         last image data

            endian = 'L';   % I suggest to use endian = 'L'; because big-endian is slower because of swapbytes.
            [~, ~, system_endian] = computer;
            if system_endian == endian
                swapbytes_flag = 0;
            else
                swapbytes_flag = 1;
            end

            %% Write Tiff Header into file
            if endian == 'L'
                fwrite(fileID, sprintf("\x49\x49\x2A\x00"), 'uint8');   % little-endian (Intel format) order
            else
                fwrite(fileID, sprintf("\x4D\x4D\x00\x2A"), 'uint8');   % big-endian (Motorola format) order
            end
            IFDOffset_uint32 = 8;
            if swapbytes_flag == 0
                IFDOffset = typecast(uint32(IFDOffset_uint32), 'uint8');
            else
                IFDOffset = typecast(swapbytes(uint32(IFDOffset_uint32)), 'uint8');
            end
            fwrite(fileID, IFDOffset, 'uint8');

            %% IFD common part
            ifd = Simple_IFD(endian);
            ifd.ImageWidth = width;
            ifd.ImageLength = height;
            ifd.RowsPerStrip = height;
            ifd = ifd.set_dtype(data_type);
            ifd.StripByteCounts = fix(width * height * ifd.BitsPerSample / 8);
            ifd.NextIFD = 0;

            %% Image description part
            hyperstack_flag = 0;
            images = channels * slices * frames;
            image_description =  sprintf("ImageJ=1.53t\nimages=%d\n", images);
            if channels > 1
                hyperstack_flag = hyperstack_flag + 1;
                image_description = image_description + sprintf("channels=%d\n", channels);
            end
            if slices > 1
                hyperstack_flag = hyperstack_flag + 1;
                image_description = image_description + sprintf("slices=%d\n", slices);
            end
            if frames > 1
                hyperstack_flag = hyperstack_flag + 1;
                image_description = image_description + sprintf("frames=%d\n", frames);
            end
            if hyperstack_flag > 1
                image_description = image_description + sprintf("hyperstack=true\n");
            end
            if channels > 1
                image_description = image_description + sprintf("mode=composite\n");
            else
                image_description = image_description + sprintf("mode=grayscale\n");
            end
            % "\u00B5" is Unicode Character 'MICRO SIGN' (U+00B5)
            % Character '\u' is not valid in Matlab, so we use '\\u00B5' or '\x00B5'
            image_description = image_description + sprintf("unit=\\u00B5m\n");
            image_description = image_description + sprintf("spacing=%0.3f\n", spacing);
            image_description = image_description + sprintf("loop=false\n");
            image_description = image_description + sprintf("min=%0.1f\n", min);
            image_description = image_description + sprintf("max=%0.1f\n", max);
            image_description = image_description + sprintf("\x00");
            ifd.ImageDescription = image_description;
            ifd.NumberCharsInImageDescription = strlength(image_description);
            ifd.OffsetOfImageDescription = IFDOffset_uint32 + size(ifd.bytes, 2);

            %% Image resolution part
            image_resolution = zeros([1, 8], 'uint8');
            
            % Old version
%             resolution_numerator = 1000000;                             % Convert mm into nm. Although we use ifd.ResolutionUnit = 3 (Centimeter), somehow ImageJ read unit from Metadata instead of ifd.ResolutionUnit.
%             resolution_denominator = uint32(1000000 * resolution);      % nm / pixel, e.g. 39 nm
            % New version: More consistent with imagej
            resolution_numerator = round(1000000 / resolution);         % Convert nm into nm.
            resolution_denominator = 1000000;                           % denominator is always 1

            if swapbytes_flag == 0
                image_resolution(1:4) = typecast(uint32(resolution_numerator), 'uint8');
                image_resolution(5:8) = typecast(uint32(resolution_denominator), 'uint8');
            else
                image_resolution(1:4) = typecast(swapbytes(uint32(resolution_numerator)), 'uint8');
                image_resolution(5:8) = typecast(swapbytes(uint32(resolution_denominator)), 'uint8');
            end
            ifd.XResolution = ifd.OffsetOfImageDescription + ifd.NumberCharsInImageDescription;
            ifd.YResolution = ifd.XResolution + 8;
            ifd.resolution =  round(single(resolution_denominator) / single(resolution_numerator), 5); % Unit: um / pixel
            

            %% Write IFD, ImageDescription and XYResolution into file
            ifd.StripOffsets = ifd.YResolution + 8;
            fwrite(fileID, ifd.bytes, 'uint8');
            fwrite(fileID, image_description, 'uint8');
            fwrite(fileID, image_resolution, 'uint8');
            fwrite(fileID, image_resolution, 'uint8');
        end

        function header = parse_tif(fileID, verbose)
            %%
            %             Open a file, determine that it's a TIF by parsing its header, and
            %             read through the TIF's Image File Directories (IFDs) one at a time
            %             to determine the structure of the TIF.
            %             See:
            %             partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
            %             for reference.
            if nargin == 1
                verbose = 0;
            elseif nargin == 2
                verbose = verbose;
            end
            header = Simple_IFD();

            [~, ~, system_endian] = computer;
            [next_ifd_offset, header.endian] = ImageJ_formatted_TIFF.parse_header(fileID, system_endian, verbose);
            header = ImageJ_formatted_TIFF.parse_ifd(fileID, header, next_ifd_offset, system_endian, verbose);
        end

        function [IFDOffset, endian] = parse_header(fileID, system_endian, verbose)
            header = ImageJ_formatted_TIFF.get_bytes_from_file(fileID, 0, 8);
            if verbose
                fprintf("Header: ");
                for i = 1:size(header, 1)
                    fprintf("%d, ", header(i));
                end
                fprintf("\n");
            end
            if header(1:4) == [73; 73; 42; 0]
                endian = "L";
            elseif header(1:4) == [77; 77; 0; 42]
                endian = "B";
            else
                ME = MException("Not a TIF file");
                throw(ME)
            end
            
            % Old version
%             IFDOffset = ImageJ_formatted_TIFF.bytes_to_int(header(5:8), endian);
            % New version
            IFDOffset = typecast(header(5:8), 'uint32');            
            if system_endian ~= endian
                IFDOffset = swapbytes(IFDOffset);
            end
            
            if verbose
                fprintf("This file is a %s-endian tif.\n", endian);
                fprintf("The offset of the first IFD is at %d bytes.\n\n", IFDOffset);
            end
        end

        function header = parse_ifd(fileID, header, ifd_offset, system_endian, verbose)
            % Old version
%             num_tags = ImageJ_formatted_TIFF.bytes_to_int(ImageJ_formatted_TIFF.get_bytes_from_file(fileID, ifd_offset, 2), header.endian);
            % New version
            num_tags = typecast(ImageJ_formatted_TIFF.get_bytes_from_file(fileID, ifd_offset, 2), 'uint16');            
            if system_endian ~= header.endian
                num_tags = swapbytes(num_tags);
            end

            if verbose
                fprintf("IFD at offset %d bytes with %d tags:\n", ifd_offset, num_tags);
            end
            header.NumDirEntries = num_tags;
            ifd_bytes = ImageJ_formatted_TIFF.get_bytes_from_file(fileID, ifd_offset + 2, 12 * num_tags + 4);
            entries = containers.Map();
            
            % The first IFD starts immediately after the summary metadata.
            % Each IFD will contain the same set of TIFF tags, except for the first one in each file,
            % which contains two ImageJ metadata tags, and two copies of the ImageDescription tag.
            % One of these contains a string needed by ImageJ to recognize these files, and the other contains OME metadata.
            ImageDescription_flag = 0;
            for i = 1:num_tags
                entry = ifd_bytes(12 * (i - 1) + 1 : 12 * i);
                if verbose
                    fprintf("   (%d) Entry:", i);
                    for e = 1:size(entry, 1)
                        fprintf(" %d,", entry(e));
                    end
                    fprintf("\n");
                end

                [tag_id, value] = ImageJ_formatted_TIFF.interpret_ifd_entry(fileID, entry, header.endian, system_endian, verbose);
                if ~isempty(value)
                    entries(tag_id) = value;
                end

                try
                    if (tag_id == "ImageDescription") && (ImageDescription_flag > 0)
                        % If we find the 2nd actual "ImageDescription", overwrite the 1st "ImageDescription" into the "OMEXMLMetadata"
                        header.NumberCharsInOMEXMLMetadata = header.NumberCharsInImageDescription;
                        header.OffsetOfOMEXMLMetadata = header.OffsetOfImageDescription;
                        header.OMEXMLMetadata = header.ImageDescription;
                    else
                        eval(sprintf("header.%s = entries('%s');", tag_id, tag_id));
                    end
                    
                    if tag_id == "ImageDescription"

                        header.NumberCharsInImageDescription = size(entries('ImageDescription'), 1);
                        % Old version
%                         header.OffsetOfImageDescription = ImageJ_formatted_TIFF.bytes_to_int(entry(9:12), header.endian);
                        % New version
                        header.OffsetOfImageDescription = typecast(entry(9:12), 'uint32');
                        if system_endian ~= header.endian
                            header.OffsetOfImageDescription = swapbytes(header.OffsetOfImageDescription);
                        end
                        header.ImageDescription = convertCharsToStrings(char(entries('ImageDescription')));
                        ImageDescription_flag = ImageDescription_flag + 1;
                    end

                    if tag_id == "XResolution"
                        % Old version
%                         resolution_numerator = ImageJ_formatted_TIFF.bytes_to_int(header.XResolution(1:4), header.endian);
%                         resolution_denominator = ImageJ_formatted_TIFF.bytes_to_int(header.XResolution(5:8), header.endian);
                        % New version
                        resolution_numerator = typecast(header.XResolution(1:4), 'uint32');
                        resolution_denominator = typecast(header.XResolution(5:8), 'uint32');
                        if system_endian ~= header.endian
                            resolution_numerator = swapbytes(resolution_numerator);
                            resolution_denominator = swapbytes(resolution_denominator);
                        end
                        if resolution_denominator == 1
                        % For some cases, tiff files do not follow unit in ImageDescription
                            header.resolution =  round(single(10000) / single(resolution_numerator), 5);                    % Unit: um / pixel
                        else                        
                            header.resolution =  round(single(resolution_denominator) / single(resolution_numerator), 5);   % Unit: um / pixel
                        end
                    end
                    
                    if tag_id == "IJMetadata"
                        header.IJMetadata = {};
                        start_index = 1;
                        for IJMetadata_index = 1:size(header.IJMetadataByteCounts, 1)
                            ByteCounts = header.IJMetadataByteCounts(IJMetadata_index);
                            header.IJMetadata{IJMetadata_index} = native2unicode(value(start_index:start_index + ByteCounts - 1), 'UTF-8');
%                             header.IJMetadata{IJMetadata_index} = convertCharsToStrings(char(value(start_index:start_index + ByteCounts - 1)));
                            start_index = start_index + ByteCounts;
                        end
                    end

                    if tag_id == "MicroManagerMetadata"
                        header.NumberCharsInMicroManagerMetadata = size(entries('MicroManagerMetadata'), 1);
                        % Old version
%                         header.OffsetOfOMicroManagerMetadata = ImageJ_formatted_TIFF.bytes_to_int(entry(9:12), header.endian);
                        % New version
                        header.OffsetOfOMicroManagerMetadata = typecast(entry(9:12), 'uint32');
                        if system_endian ~= header.endian
                            header.OffsetOfOMicroManagerMetadata = swapbytes(header.OffsetOfOMicroManagerMetadata);
                        end
                        header.MicroManagerMetadata = convertCharsToStrings(char(entries('MicroManagerMetadata')));
                        if verbose
                            fprintf("\n");
                        end
                    end
                catch
                    if verbose
                        warning("Simple_IFD class does not contain this Tiff tag: %d\n", tag_id);
                    end
                end
            end
            
            % Try to find "images", "channels", "slices", "frames", "unit" and "spacing" from the (2nd) ImageDescription
            if ~isempty(header.ImageDescription)
                ImageDescription_split = split(header.ImageDescription, newline);

                % Check "channels"
                if sum(contains(ImageDescription_split, 'channels'))
                    index = find(contains(ImageDescription_split, 'channels'), 1, 'first');
                    header.channels = sscanf(ImageDescription_split(index), 'channels=%d');
                end 

                % Check "slices"
                if sum(contains(ImageDescription_split, 'slices'))
                    index = find(contains(ImageDescription_split, 'slices'), 1, 'first');
                    header.slices = sscanf(ImageDescription_split(index), 'slices=%d');
                end

                % Check "frames"
                if sum(contains(ImageDescription_split, 'frames'))
                    index = find(contains(ImageDescription_split, 'frames'), 1, 'first');
                    header.frames = sscanf(ImageDescription_split(index), 'frames=%d');
                end
                
                % Check "unit"
                if sum(contains(ImageDescription_split, 'unit'))
                    index = find(contains(ImageDescription_split, 'unit'), 1, 'first');
                    unit = sscanf(ImageDescription_split(index), 'unit=%s');
                    switch unit
                        case {'um', 'micron', '\u00B5m'}
                            header.unit = 'um';
                        otherwise
                            header.unit = unit;
                    end
                end

                % Check "spacing"
                if sum(contains(ImageDescription_split, 'spacing'))
                    index = find(contains(ImageDescription_split, 'spacing'), 1, 'first');
                    header.spacing = sscanf(ImageDescription_split(index), 'spacing=%f');
                end

                % Finally check "images"
                if sum(contains(ImageDescription_split, 'images'))
                    index = find(contains(ImageDescription_split, 'images'), 1, 'first');
                    header.images = sscanf(ImageDescription_split(index), 'images=%d');
                elseif ~isempty(header.channels) || ~isempty(header.slices) || ~isempty(header.frames)
                    header.images = 1;
                    if ~isempty(header.channels)
                        header.images = header.images * header.channels;
                    end
                    if ~isempty(header.slices)
                        header.images = header.images * header.slices;
                    end
                    if ~isempty(header.frames)
                        header.images = header.images * header.frames;
                    end
                end 
            end
        end

        function [tag_id, value] = interpret_ifd_entry(fileID, entry, endian, system_endian, verbose)
            % Old version
%             tag_id = ImageJ_formatted_TIFF.bytes_to_int(entry(1:2), endian);
            % New version
            tag_id = typecast(entry(1:2), 'uint16');
            if system_endian ~= endian
                tag_id = swapbytes(tag_id);
            end

            keySet = {254, 256, 257, 258, 259, 262, 270, 273, 277, 278, 279, 282, 283, 296, 339, 50838, 50839, 51123};
            valueSet = {'NewSubFileType', 'ImageWidth', 'ImageLength', 'BitsPerSample', 'Compression', 'PhotometricInterpretation', 'ImageDescription', 'StripOffsets', 'SamplesPerPixel', 'RowsPerStrip', 'StripByteCounts', 'XResolution', 'YResolution', 'ResolutionUnit', 'SampleFormat', 'IJMetadataByteCounts', 'IJMetadata', 'MicroManagerMetadata'};
            tag_id_lookup = containers.Map(keySet, valueSet);
            try
                tag_id = tag_id_lookup(tag_id);

                % Old version
%                 data_type = ImageJ_formatted_TIFF.bytes_to_int(entry(3:4), endian);
                % New version
                data_type = typecast(entry(3:4), 'uint16');
                if system_endian ~= endian
                    data_type = swapbytes(data_type);
                end

                keySet = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                valueSet = {{'BYTE', 1}, {'ASCII', 1}, {'SHORT', 2}, {'LONG', 4}, {'RATIONAL', 8}, {'SBYTE', 1}, {'UNDEFINED', 8}, {'SSHORT', 2}, {'SLONG', 4}, {'SRATIONAL', 8}, {'FLOAT', 4}, {'DOUBLE', 8}};
                data_type_lookup = containers.Map(keySet, valueSet);
                try
                    value = data_type_lookup(data_type);
                    data_type = string(value(1));
                    bytes_per_count = cell2mat(value(2));
                catch ME
                    warning("Unknown data type in TIF tag: %s", data_type);
                    throw(ME);
                end
                
                % Old version
%                 data_count = ImageJ_formatted_TIFF.bytes_to_int(entry(5:8), endian);
                % New version
                data_count = typecast(entry(5:8), 'uint32');
                if system_endian ~= endian
                    data_count = swapbytes(data_count);
                end

                value_size_bytes = double(data_count) * double(bytes_per_count);
                if value_size_bytes <= 4
                    % The DataOffset directly encode the value
                    value = entry(9 : 8 + value_size_bytes);
                else
                    % The DataOffset encodes a pointer to the value
                    % Old version
%                     offset = ImageJ_formatted_TIFF.bytes_to_int(entry(9:12), endian);
                    % New version
                    offset = typecast(entry(9:12), 'uint32');
                    if system_endian ~= endian
                        offset = swapbytes(offset);
                    end
                    value = ImageJ_formatted_TIFF.get_bytes_from_file(fileID, offset, value_size_bytes);
                end

                % We still haven't converted the value from bytes yet, but at least we
                % got the correct bytes that encode the value.
                switch data_type
                    case 'BYTE'
                        if verbose
                            fprintf("	%s:\n", tag_id);
%                             content = convertCharsToStrings(char(value));
%                             fprintf("	%s:\n%s", tag_id, content);
                        end

                    case 'ASCII'
                        if verbose
                            content = convertCharsToStrings(char(value));
                            fprintf("	%s:\n%s\n", tag_id, content);

                        end
                         
                    case 'SHORT'
                        value = typecast(value, 'uint16');                        
                        if system_endian ~= endian
                            value = swapbytes(value);
                        end
                        
                        if verbose
                            if data_count == 1                            
                                fprintf("	%s: %d\n", tag_id, value);
                            else
                                fprintf(['	%s: [', repmat('%d, ', 1, numel(value) - 1), '%d]\n'], tag_id, value);
                            end
                        end

                    case 'LONG'
                        value = typecast(value, 'uint32');                        
                        if system_endian ~= endian
                            value = swapbytes(value);
                        end
                        
                        if verbose
                            if data_count == 1                            
                                fprintf("	%s: %d\n", tag_id, value);
                            else
                                fprintf(['	%s: [', repmat('%d, ', 1, numel(value) - 1), '%d]\n'], tag_id, value);
                            end
                        end
                   
                    case 'RATIONAL'
                        if verbose
                            fprintf("	%s:", tag_id);
                            for v = 1:size(value, 1)
                                fprintf(" %d,", value(v));
                            end
                            fprintf("\n");
                        end
                end
            catch
                if verbose
                    % Old version
%                     warning("Unknown tag ID in TIF: %d with value / offset: %d\n", tag_id, ImageJ_formatted_TIFF.bytes_to_int(entry(9:12), endian));
                    % New version
                    value = typecast(entry(9:12), 'uint32');
                    if system_endian ~= endian
                        value = swapbytes(value);
                    end
                    warning("Unknown tag ID in TIF: %d with value / offset: %d\n", tag_id, value);
                end
                value = [];
            end
            
        end

        function bytes = get_bytes_from_file(fileID, offset, num_bytes)
            fseek(fileID, offset, 'bof');
            bytes = fread(fileID, num_bytes, 'uint8=>uint8');
        end

        function value = bytes_to_int(bytes, endian)
            value = 0;
            if endian == 'L'
                for i = 1:size(bytes, 1)
                    value = value + bytes(i) * 256 ^ (i - 1);
                end
            elseif endian == 'B'
                for i = 1:size(bytes, 1)
                    value = value + bytes(i) * 256 ^ (size(bytes, 1) - i);
                end
            else
                ME = MException("'endian' must be either big or little");
                throw(ME)
            end
        end

    end
end

%% Test of write image stack
%     stack_in = uint16(linspace(0, 65535, 1920 * 1600));
%     stack_in = repmat(stack_in, 1, 1, 500);
%     stack_in = reshape(stack_in, [1920, 1600, 500]);
%     stack_in = permute(stack_in, [2 1 3]);
%     tic
%     fileID = fopen('test.tif', 'w+');
%     ImageJ_formatted_TIFF.write_IFD(fileID, 1920, 1600, class(stack_in), 1, 500, 1, 0.1, 0, 65535, 0.05);
%     fwrite(fileID, stack_in, class(stack_in));%
%     fclose(fileID);
%     toc

%% Test of read image stack
%     fileID = fopen('test.tif', 'r');
%     header = ImageJ_formatted_TIFF.parse_tif(fileID);
%     tic
%     fseek(fileID, header.StripOffsets, 'bof');
%     matrix_out = fread(fileID, [1, 1920 * 1600], 'uint16');
%     fclose(fileID);
%     toc