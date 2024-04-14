<script lang="ts">
import { defineComponent, ref } from 'vue';

const props = defineProps()

export default defineComponent({
    setup() {
        const hello = ref(1)

        return {
            hello
        }
    }
})
</script>

<template>
    <div>
        hello
        {{ hello }}
    </div>
</template>
