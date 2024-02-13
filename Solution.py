class Solution:
	def isPalindrome(self, x: int) -> bool:
		x = str(x)
		if x[::-1] == x:
			return True
		return False

	def twoSum(self, nums: list[int], target: int) -> list[int]:
		answer = []
		ind1 = 0
		ind2 = 0
		for i in range(len(nums) - 1):
			for j in range(i + 1, len(nums)):
				if nums[i] + nums[j] == target:
					answer.append(i)
					answer.append(j)
					return answer

		return answer


# if __name__ == '__main__':
# 	s = Solution()
# 	res = s.twoSum([3,3], 6)
# 	print(res)