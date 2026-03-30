import numpy as np

def inspect_npy(path):
    data = np.load(path, allow_pickle=True)

    print("=" * 50)
    print(f"File: {path}")
    print("=" * 50)

    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Size (total elements):", data.size)

    if data.size > 0:
        print("First element:", data.flatten()[0])

    # 数值类型统计
    if np.issubdtype(data.dtype, np.number):
        print("\n[Statistics]")
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())
        print("Std:", data.std())

        # 检查是否有 nan/inf
        print("Has NaN:", np.isnan(data).any())
        print("Has Inf:", np.isinf(data).any())

    # 如果是 object（很多事件数据会这样）
    else:
        print("\n[Object Data Preview]")
        try:
            print("Sample:", data[:3])
        except:
            print("Cannot slice preview")
    print("polarity unique:", np.unique(data[:, 0]))
    print("=" * 50)




# 使用
inspect_npy("/Users/zwj/Documents/毕设/EMRS-BAIDU/val/events/low_light/0001_low_light/0001_low_light_1698324603563418.npy")