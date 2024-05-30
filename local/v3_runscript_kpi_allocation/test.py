from multiprocessing import cpu_count, Pool, Manager
import time


def worker(result, number, dependency=None):
    # Đợi sự kiện từ tiến trình phụ thuộc
    if dependency:
        dependency.wait()

    # Thực hiện tính toán
    result[number] = number * 2


def main():
    num_processes = cpu_count()
    num_tasks = 5

    # Tạo một quản lý đối tượng cho danh sách kết quả
    with Manager() as manager:
        result = manager.list([None] * num_tasks)

        # Sử dụng Pool để quản lý các tiến trình
        with Pool(processes=num_processes) as pool:
            for i in range(num_tasks):
                dependency = result[i-1] if i > 0 else None
                pool.apply_async(worker, args=(result, i, dependency))

            # Chờ tất cả các tiến trình hoàn thành
            pool.close()
            pool.join()

            # In kết quả
            for i, r in enumerate(result):
                print(f"Task {i}: Result = {r}")


if __name__ == "__main__":
    main()
