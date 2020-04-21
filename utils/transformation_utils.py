import torch


def transform_voxel_to_match_image(voxel):
    # in: bsz, channels, height, width, depth
    # out: bsz, channels, width, height, depth
    voxel = voxel.permute(0, 1, 3, 2, 4)
    # for camera mirror flip across width
    voxel = voxel.flip(2)

    return voxel


def generate_transform_matrix(transform_params):
    # bsz = transform_params.size(0)
    Ry = rad2Ry(transform_params[:, 0])
    Rx = rad2Rx(transform_params[:, 1])
    # Rz = rad2Ry(transform_params[:, 0])
    S = scale2S(transform_params[:, 3:6])
    T = translation2T(transform_params[:, 6:])

    # transforms are performed from left to right
    # first rotation, then scale and finally translation
    A = torch.bmm(torch.bmm(torch.bmm(Ry, Rx), S), T)
    return A


def generate_transform_matrix_with_recentering(
    transform_params, in_size=64, out_size=64
):
    bsz = transform_params.size(0)
    Ry = rad2Ry(transform_params[:, 0])
    Rx = rad2Rx(transform_params[:, 1])
    # Rz = rad2Ry(transform_params[:, 0])
    S = scale2S(transform_params[:, 3:6])
    T = translation2T(transform_params[:, 6:])

    # Move origin to old grid center translation
    # This is to rotate the objects around it's origin
    CO = (
        translation2T(torch.tensor([[-in_size * 0.5, -in_size * 0.5, -in_size * 0.5]]))
        .repeat(bsz, 1, 1)
        .to(transform_params.device)
    )
    # Move origin to new grid border
    DN = (
        translation2T(torch.tensor([[out_size * 0.5, out_size * 0.5, out_size * 0.5]]))
        .repeat(bsz, 1, 1)
        .to(transform_params.device)
    )

    # DN * T * S * Rx * Ry * CO
    A = torch.bmm(torch.bmm(torch.bmm(torch.bmm(torch.bmm(DN, T), S), Rx), Ry), CO)
    return A


def generate_inv_transform_matrix(
    transform_params, in_size=64, out_size=64,
):
    bsz = transform_params.size(0)
    Ry_inv = invR(rad2Ry(transform_params[:, 0]))
    Rx_inv = invR(rad2Ry(transform_params[:, 1]))
    # Rz = rad2Ry(transform_params[:, 0])
    S_inv = invS(scale2S(transform_params[:, 3:6]))
    T_inv = invT(translation2T(transform_params[:, 6:]))

    # Move origin to old grid center translation
    # This is to rotate the objects around it's origin
    CO_inv = (
        invT(
            translation2T(
                torch.tensor([[-in_size * 0.5, -in_size * 0.5, -in_size * 0.5]])
            )
        )
        .repeat(bsz, 1, 1)
        .to(transform_params.device)
    )
    # Move origin to new grid border
    DN_inv = (
        invT(
            translation2T(
                torch.tensor([[out_size * 0.5, out_size * 0.5, out_size * 0.5]])
            )
        )
        .repeat(bsz, 1, 1)
        .to(transform_params.device)
    )

    # CO^-1 * Ry^-1 * Rx^-1 * S^-1 * T^-1 * DN^-1
    A = torch.bmm(
        torch.bmm(
            torch.bmm(torch.bmm(torch.bmm(CO_inv, Ry_inv), Rx_inv), S_inv), T_inv
        ),
        DN_inv,
    )
    return A


def rad2Ry(radians, requires_grad=False):
    bsz = radians.size(0)
    radians = radians.view(bsz)
    R = torch.eye(4).unsqueeze(0).repeat(bsz, 1, 1)
    R[:, 0, 0] = torch.cos(radians)
    R[:, 0, 2] = -torch.sin(radians)
    R[:, 2, 0] = torch.sin(radians)
    R[:, 2, 2] = torch.cos(radians)

    return R.to(radians.device)


def rad2Rx(radians, requires_grad=False):
    bsz = radians.size(0)
    radians = radians.view(bsz)
    R = torch.eye(4).unsqueeze(0).repeat(bsz, 1, 1)
    R[:, 0, 0] = torch.cos(radians)
    R[:, 0, 1] = torch.sin(radians)
    R[:, 1, 0] = -torch.sin(radians)
    R[:, 1, 1] = torch.cos(radians)

    return R.to(radians.device)


def invR(R):
    return R.transpose(1, 2)


def translation2T(translation, requires_grad=False):
    bsz = translation.size(0)
    T = torch.eye(4).unsqueeze(0).repeat(bsz, 1, 1)
    T[:, :3, 3] = translation

    return T.to(translation.device)


def invT(T):
    T_inv = T.clone()
    T_inv[:, :3, 3] = -T_inv[:, :3, 3]
    return T_inv


def scale2S(scale):
    bsz = scale.size(0)
    S = torch.eye(4).unsqueeze(0).repeat(bsz, 1, 1)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]

    return S.to(scale.device)


def invS(S):
    r = 1 / S.diagonal(dim1=1, dim2=2)
    S_inv = S.clone()
    S_inv[:, 0, 0] = r[:, 0]
    S_inv[:, 1, 1] = r[:, 1]
    S_inv[:, 2, 2] = r[:, 2]
    return S_inv
