opt = globals();
pascal_init;

voc_dir = '/Users/JudyYe/Documents/CMUNewBorn/0_research/imp_shape/PASCAL3D+_release1.1/PASCAL/VOCdevkit';
p3d_dir = '../';

basedir = fullfile(pwd, '..', '..');
seg_kp_dir = fullfile(basedir, 'cachedir', 'pascal', 'segkps');

img_anno_dir = fullfile(basedir, 'cachedir', 'p3d', 'data');
sfm_anno_dir = fullfile(basedir, 'cachedir', 'p3d', 'sfm');

if ~ exist(img_anno_dir)
  mkdir(img_anno_dir);
end;

if ~ exist(sfm_anno_dir)
  mkdir(sfm_anno_dir);
end;

% categories = {'aeroplane', 'car'};
categories = {'chair'}; % , 'aeroplane', 'car'};
% categories = {'n02114855', 'n02126139', 'n02381460', 'n02390640', 'n02390738', 'n02391049', 'n02391234', ...
%                       'n02391373', 'n02391994', 'n02393580', 'n02393940', 'n02397096', 'n02402425', 'n02408429', ...
%                       'n02410702', 'n02411705', 'n02416519', 'n02417070', 'n02421449', 'n02422106', 'n02423589', ...
%                       'n02426813', 'n02428349', 'n02430045', 'n02437616', 'n02437971', 'n02439033'};
% categories = {'n02126139'};
for c = 1:length(categories)
    category = categories{c};

    % if exist(fullfile(sfm_anno_dir, [category '_val.mat']))
        % continue
    % end
    disp(category);
    [class_data_pascal] = extract_class_data_p3d_old(category, p3d_dir, voc_dir, 1, seg_kp_dir);
    [class_data_imgnet] = extract_class_data_p3d_old(category, p3d_dir, voc_dir, 0, seg_kp_dir);
    % class_data = class_data_imgnet;
    class_data = [class_data_pascal class_data_imgnet];
    fprintf('data %d %d %d\n', length(class_data), length(class_data_imgnet), length(class_data_pascal));

    % horz_edges should be front to back
    % if strcmp(category, 'car')
    %     horz_edges = [2 4; 6 8];
    % elseif strcmp(category, 'aeroplane')
    %     horz_edges = [8 3];
    % else
    %     disp('Data not available');
    %     keyboard;
    % end
    % [sfm_anno, sfm_verts, sfm_faces, kp_perm_inds] = pascal_sfm(class_data, kp_names, horz_edges, []);

    % good_inds = ([sfm_anno.err_sfm_reproj] < 0.01);
    % class_data = class_data(good_inds);
    % sfm_anno = sfm_anno(good_inds);

    train_ids = [class_data.is_train]; train_ids = (train_ids==1);
    % all_sfm_struct = struct('sfm_anno', sfm_anno, 'S', sfm_verts, 'conv_tri', sfm_faces);
    % train_sfm_struct = struct('sfm_anno', sfm_anno(train_ids), 'S', sfm_verts, 'conv_tri', sfm_faces);
    % val_sfm_struct = struct('sfm_anno', sfm_anno(~train_ids), 'S', sfm_verts, 'conv_tri', sfm_faces);
    
    % save(fullfile(img_anno_dir, [category '_kps']), 'kp_names', 'kp_perm_inds');


    % if strcmp(category, 'car') == 1

    %     all_img_struct = struct('images', class_data);
    %     train_img_struct = struct('images', class_data(train_ids));
    %     val_img_struct = struct('images', class_data(~train_ids));

    %     % chunk = 3
    %     mid_all = round(length(all_img_struct.images) / 2);
    %     mid_train = round(length(train_img_struct.images) / 2);
    %     mid_val = round(length(val_img_struct.images) / 2);

    %     all_img_struct1 = struct('images', all_img_struct.images(1: mid_all));
    %     train_img_struct1 = struct('images', train_img_struct.images(1: mid_train));
    %     val_img_struct1 = struct('images', val_img_struct.images(1 : mid_val));

    %     all_img_struct2 = struct('images', all_img_struct.images(1 + mid_all: end));
    %     train_img_struct2 = struct('images', train_img_struct.images(1 + mid_train : end));
    %     val_img_struct2 = struct('images', val_img_struct.images(1 + mid_val: end));

    %     for i = 1 : 2
    %         % save(fullfile(img_anno_dir, [category '_all' int2str(i)]), '-struct', ['all_img_struct' int2str(i)]);
    %         save(fullfile(img_anno_dir, [category '_train' int2str(i)]), '-struct', ['train_img_struct' int2str(i)]);
    %         save(fullfile(img_anno_dir, [category '_val' int2str(i)]), '-struct', ['val_img_struct' int2str(i)]);
    %     end
    % else
    all_img_struct = struct('images', class_data);
    train_img_struct = struct('images', class_data(train_ids));
    val_img_struct = struct('images', class_data(~train_ids));
    chunk = 1
    fprintf('%d %d', length(all_img_struct.images), length(train_img_struct.images));
    save(fullfile(img_anno_dir, [category '_all'   ]), '-struct', 'all_img_struct');
    save(fullfile(img_anno_dir, [category '_train' ]), '-struct', 'train_img_struct');
    save(fullfile(img_anno_dir, [category '_val'   ]), '-struct', 'val_img_struct');
    % end


    % save(fullfile(sfm_anno_dir, [category '_all']), '-struct', 'all_sfm_struct');
    % save(fullfile(sfm_anno_dir, [category '_train']), '-struct', 'train_sfm_struct');
    % save(fullfile(sfm_anno_dir, [category '_val']), '-struct', 'val_sfm_struct');

end

