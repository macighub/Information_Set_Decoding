import math

import numpy as np

def pack2uint64(data):
    """
    Converts rows of a matrix or a vector into bit-packed uint64 format
    with an extra sentinel bit 1, which marks the end of useful bits and
    allows extraction of useful bits. Last uint64 unit is right-padded
    with 0 up to 64 bits.

    Parameters:
    data: binary matrix or vector

    Returns:
    data_packed (ndarray): matrix or vector in bit-packed uint64 format
                           with included sentinel bit which marks the
                           end of useful bits.
    """
    data_packed = []

    if data.ndim == 2: # Data is a matrix -> pack each row into uint64 units
        for row_cnt in range(data.shape[0]):
            data_packed.append(pack2uint64(data[row_cnt]))
    else: # Data is a vector -> pack it into uint64 units
        # Calculate number of uint64 units
        # (includes all elements, extra '1' bit
        # and padded zeros up to multiple of 64).
        num_units = math.ceil((data.shape[0] + 1) / 64)

        # Append the sentinel '1' at the end
        data_bits = np.append(data, 1)

        data_packed = np.zeros(num_units, dtype=np.uint64)

        for unit_cnt in range(num_units):
            start_pos, end_pos = unit_cnt * 64, min((unit_cnt + 1) * 64, len(data_bits))
            unit_bits = data_bits[start_pos:end_pos]

            if unit_bits.size > 0:
                data_packed[unit_cnt] = np.packbits(np.pad(unit_bits,
                                                           (0, 64 - len(unit_bits)),
                                                           constant_values=0),
                                                    bitorder='little').view(np.uint64)[0]

    return np.array(data_packed)

def packed_uint64_length(data):
    """
    Extracts number of useful bits from a bit-packed uint64
    format of a matrix or a vector using encoded sentinel bit.

    Parameters:
    data: bit-packed uint64 matrix or vector

    Returns:
    sentinel_pos (uint64): index of sentinel bit in the full
                           binary format of data.
    """
    sentinel_pos = 0
    full_units = 0
    last_unit = 0

    if data.ndim > 1:
        # Data is a matrix -> The sentinel bit position's highest value is extracted recursively.
        for row_cnt in range(data.shape[0]):
            sentinel_pos = max(sentinel_pos, packed_uint64_length(data[row_cnt]))
    else:
        match data.ndim:
            case 0:
                # Data is one uint64 unit -> The position of the sentinel bit is extracted.
                last_unit = int(data)
            case 1:
                # Data is a vector of uint64 units -> The total bits in all uint64 units, except
                # the last uint64 unit + position of the sentinel bit in the last uint64 unit is
                # extracted.
                full_units = len(data) - 1
                last_unit = int(data[-1])

        sentinel_pos = np.uint64(full_units * 64) + np.uint64(last_unit.bit_length() - 1)

    return sentinel_pos

def popcount_uint64(data):
    """
    Efficiently computes the population count (number of set bits) in a bit-packed uint64 format
    of a vector or rows of a matrix.

    This method uses a series of bitwise operations known as the "Hacker's Delight" algorithm to
    quickly count the number of 1-bits (also known as the Hamming weight) in each uint64 integer:

    1. arr - ((arr >> 1) & 0x5555555555555555):
       Counts bits in pairs, subtracting bits shifted right by 1 and masked.

    2. (arr & 0x3333333333333333) + ((arr >> 2) & 0x3333333333333333):
       Counts bits in quartets by adding two-bit counts from previous step.

    3. ((arr + (arr >> 4)) & 0x0F0F0F0F0F0F0F0F):
       Sums counts within each nibble (4 bits).

    4. Multiplying by 0x0101010101010101 and shifting by 56:
       Sums all counts within each byte to produce the final count.
    """
    data = np.asarray(data, dtype=np.uint64)
    data = data - ((data >> 1) & 0x5555555555555555)
    data = (data & 0x3333333333333333) + ((data >> 2) & 0x3333333333333333)
    data = (data + (data >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((data * 0x0101010101010101) >> 56) & 0x7F

def clear_sentinel_bit(data):
    """Clears the sentinel bit (rightmost set bit) from last uint64 value."""
    data_cleared = []

    if data.ndim == 2: # Data is a matrix -> clear sentinel bit from each row of uint64 units
        for row_cnt in range(data.shape[0]):
            data_cleared.append(clear_sentinel_bit(data[row_cnt]))
    else: # Data is a vector -> clear sentinel bit
        # Copy to avoid changing original
        data_cleared = data.copy()

        # Process only the last uint64 unit (assuming sentinel appears there)
        last_unit = data_cleared[-1]

        # Clear sentinel bit
        data_cleared[-1] = last_unit & ~(np.uint64(1) << packed_uint64_length(last_unit))

    return np.array(data_cleared)


def bitpacked_dot_row_optimized(H, m):
    """
    Optimized dot product between bit-packed matrix H and vector m,
    each stored in bit-packed uint64 format.
    """
    # Clear sentinel bits
    H_clean = clear_sentinel_bit(H)
    m_clean = clear_sentinel_bit(m)

    # Perform bitwise AND between each row of the matrix and vector
    and_result = H_clean & m_clean
    #Count bits of value 1 in each uint64 units of rows of and_result
    bit_counts = popcount_uint64(and_result)

    # Sum popcounts for each row, producing the final dot product
    result = bit_counts.sum(axis=1)
    return result


def bitpacked_dot_row_readable(H, m):
    """
    Readable (but slower) version of dot product by converting matrix and vector back
    to binary format using np.unpackbits and performing dot product between them.
    """

    # Clear sentinel bits
    H_clean = clear_sentinel_bit(H)
    m_clean = clear_sentinel_bit(m)

    # Unpack bits from H and m
    H_bits = np.unpackbits(H_clean.view(np.uint8), axis=1,  bitorder='little')
    m_bits = np.unpackbits(m_clean.view(np.uint8), bitorder='little')

    # Perform the dot product directly
    result = np.dot(H_bits, m_bits)

    return result

def bitpacked_dot_column_optimized(H, y, num_columns=0, with_idx=False):
    """
    Efficiently computes the dot product between numeric array `y` (shape k)
    and each column of binary bit-packed matrix `H` (shape k x units_per_row).
    Only considers columns up to num_columns (excluding sentinel).

    :param y: np.array, dtype=np.uint64, shape=(k,)
    :param H: np.array, dtype=np.uint64, shape=(k, units_per_row)
    :param num_columns: int, total columns excluding sentinel bit
    :return: np.array, dtype=np.uint64, shape=(num_columns,), dot product per column
    """
    # Extract number of useful columns
    if num_columns == 0:
        num_columns = packed_uint64_length(H)

    result = np.zeros(num_columns, dtype=np.uint64)

    # Iterate through each binary column
    for col_idx in range(num_columns):
        # Determine which uint64 unit and which bit within that unit
        unit_idx = col_idx // 64
        bit_idx = col_idx % 64

        # Extract bits from all rows simultaneously
        column_bits = (H[:, unit_idx] >> bit_idx) & 1

        # Compute dot product with y efficiently
        result[col_idx] = np.dot(y, column_bits)

    if with_idx:
        # Convert result to a list of tuples with column indices
        return [(col_idx, result[col_idx]) for col_idx in range(num_columns)]
    else:
        return result