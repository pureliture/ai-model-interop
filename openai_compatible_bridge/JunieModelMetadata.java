import com.intellij.ml.llm.matterhorn.llm.LLM;
import java.lang.reflect.Field;
import java.util.List;

/** Reads metadata from the installed Junie version, without contacting a provider. */
class JunieModelMetadata {
    public static void main(String[] args) throws Exception {
        Field field = LLM.class.getDeclaredField("knownModels");
        field.setAccessible(true);
        for (Object value : (List<?>) field.get(null)) {
            LLM model = (LLM) value;
            var capabilities = model.getCapabilities();
            Integer context = capabilities.getMaxContextTokens();
            Integer output = capabilities.getMaxOutputTokens();
            if (context != null || output != null) {
                System.out.println(model.getName() + "\t" + (context == null ? "" : context)
                        + "\t" + (output == null ? "" : output));
            }
        }
    }
}
